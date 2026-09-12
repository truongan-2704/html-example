"""Export untrained Orthogonal Efficient through YOLO API and check ONNX CPU runtime.

Also export raw one-to-one logits to compare numerically without unstable top-k ties
from random untrained predictions. Does not install packages or evaluate accuracy.
"""
from __future__ import annotations

from copy import deepcopy
import argparse
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ['YOLO_AUTOINSTALL']='false'
config_dir=ROOT/'experiments/orthogonal_audit/ultralytics_settings'
config_dir.mkdir(parents=True,exist_ok=True)
os.environ['YOLO_CONFIG_DIR']=str(config_dir)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selected',action='store_true',help='Validate Efficient + selected decoding, including a raw-logit decoder oracle')
    parser.add_argument('--rank',action='store_true',help='Validate the actual Orthogonal-Rank B+C architecture')
    parser.add_argument('--capacity',action='store_true',help='Validate the full-spatial capacity-first architecture')
    args=parser.parse_args()
    if sum((args.selected,args.rank,args.capacity))>1:
        parser.error('Choose only one architecture probe')
    import numpy as np
    import onnx
    import onnxruntime as ort
    import torch
    from ultralytics import YOLO

    torch.set_num_threads(1)
    torch.manual_seed(17)
    out=ROOT/'experiments'/('orthogonal_capacity' if args.capacity else 'orthogonal_rank' if args.rank else 'orthogonal_selected' if args.selected else 'orthogonal_audit')
    out.mkdir(parents=True,exist_ok=True)
    ref=torch.load(ROOT/'experiments/orthogonal_audit/before_changes.pt',map_location='cpu',weights_only=True)
    config='yolo11-orthogonal-capacity.yaml' if args.capacity else 'yolo11-orthogonal-rank-bc.yaml' if args.rank else 'yolo11-orthogonal-efficient-sd.yaml' if args.selected else 'yolo11-orthogonal-efficient.yaml'
    config_root=ROOT/'configs/research/orthogonal_rank' if args.rank else ROOT/'ultralytics/cfg/models/11/yolo11-Orthogonal'
    model=YOLO(str(config_root/config))
    if args.rank:
        from ultralytics.nn.tasks import DetectionModel
        from ultralytics.nn.modules.orthogonal_rank import transfer_orthogonal_rank
        source=DetectionModel(deepcopy(ref['config']),verbose=False).eval()
        source.load_state_dict(ref['state_dict'],strict=True)
        transfer_orthogonal_rank(source,model.model)
    elif not args.capacity:
        model.model.load_state_dict(ref['state_dict'],strict=True)
    checkpoint=out/('capacity_untrained.pt' if args.capacity else 'rank_untrained.pt' if args.rank else 'efficient_untrained.pt')
    model.save(checkpoint)
    model=YOLO(checkpoint)
    exported=model.export(format='onnx',imgsz=128,batch=1,dynamic=True,simplify=False,opset=17,device='cpu',max_det=100)
    onnx.checker.check_model(onnx.load(exported))
    options=ort.SessionOptions()
    options.intra_op_num_threads=1
    options.inter_op_num_threads=1
    session=ort.InferenceSession(str(exported),sess_options=options,providers=['CPUExecutionProvider'])
    shapes=[(1,3,128,160),(2,3,160,192)]
    checks=[]
    for shape in shapes:
        x=torch.randn(*shape)
        pred=session.run(None,{session.get_inputs()[0].name:x.numpy()})[0]
        assert pred.shape==(shape[0],100,6) and np.isfinite(pred).all()
        checks.append(dict(input=list(shape),output=list(pred.shape),finite=True))

    class RawHead(torch.nn.Module):
        def __init__(self,m):
            super().__init__()
            self.m=m

        def forward(self,x):
            preds=self.m(x)[1]['one2one']
            return preds['boxes'],preds['scores']

    raw=RawHead(deepcopy(model.model).float().eval().fuse(verbose=False)).eval()
    raw_path=out/('capacity_raw_logits.onnx' if args.capacity else 'rank_raw_logits.onnx' if args.rank else 'efficient_raw_logits.onnx')
    with torch.inference_mode():
        torch.onnx.export(raw,torch.randn(*shapes[0]),str(raw_path),opset_version=17,
                          input_names=['images'],output_names=['boxes','scores'],
                          dynamic_axes={'images':{0:'batch',2:'height',3:'width'},'boxes':{0:'batch',2:'anchors'},'scores':{0:'batch',2:'anchors'}})
    onnx.checker.check_model(onnx.load(raw_path))
    session=ort.InferenceSession(str(raw_path),sess_options=options,providers=['CPUExecutionProvider'])
    for check,shape in zip(checks,shapes):
        x=torch.randn(*shape)
        with torch.inference_mode():
            expected=raw(x)
        actual=session.run(None,{'images':x.numpy()})
        errors=[]
        for a,b in zip(actual,expected):
            np.testing.assert_allclose(a,b.numpy(),rtol=1e-4,atol=1e-4)
            errors.append(float(np.abs(a-b.numpy()).max()))
        check['raw_logits_max_abs_errors']=errors
    result=dict(onnx=onnx.__version__,onnxruntime=ort.__version__,provider='CPUExecutionProvider',opset=17,
                checks=checks,trained=False,numerical_scope='Raw logits compared at 1e-4 tolerance. Official post-topk outputs checked for shape/finite values only; tied random predictions may rank differently across backends.')
    result['config']=config
    if args.selected:
        from ultralytics.nn.modules.selected_decode import SelectDecodeDetect

        class SelectedRaw(torch.nn.Module):
            def __init__(self,agnostic):
                super().__init__()
                self.head=SelectDecodeDetect(4,16,True,(8,16,32)).eval()
                self.head.stride=torch.tensor([8.,16.,32.])
                self.head.max_det=100
                self.head.dynamic=True
                self.head.export=True
                self.head.agnostic_nms=agnostic

            def predictions(self,images,boxes,scores):
                return dict(feats=[images[:,:1,::s,::s] for s in (8,16,32)],boxes=boxes,scores=scores)

            def forward(self,images,boxes,scores):
                return self.head._selected_inference(self.predictions(images,boxes,scores))

        def inputs(shape):
            b,_,h,w=shape
            n=sum((h//s)*(w//s) for s in (8,16,32))
            # Separated class scores avoid cross-backend top-k ties in this oracle.
            scores=torch.linspace(-4,4,b*4*n)[torch.randperm(b*4*n)].reshape(b,4,n)
            return torch.randn(*shape),torch.randn(b,64,n)*2,scores

        decoder_checks=[]
        for agnostic in (False,True):
            wrapper=SelectedRaw(agnostic).eval()
            path=out/('selected_decoder_agnostic.onnx' if agnostic else 'selected_decoder_classaware.onnx')
            with torch.inference_mode():
                torch.onnx.export(wrapper,inputs(shapes[0]),str(path),opset_version=17,
                                  input_names=['images','boxes','scores'],output_names=['detections'],
                                  dynamic_axes={'images':{0:'batch',2:'height',3:'width'},'boxes':{0:'batch',2:'positions'},
                                                'scores':{0:'batch',2:'positions'},'detections':{0:'batch'}})
            onnx.checker.check_model(onnx.load(path))
            session=ort.InferenceSession(str(path),sess_options=options,providers=['CPUExecutionProvider'])
            for shape in shapes:
                values=inputs(shape)
                with torch.inference_mode():
                    raw_preds=wrapper.predictions(*values)
                    expected=wrapper.head.postprocess(wrapper.head._inference(raw_preds).permute(0,2,1)).numpy()
                actual=session.run(None,{k:v.numpy() for k,v in zip(('images','boxes','scores'),values)})[0]
                np.testing.assert_allclose(actual,expected,rtol=1e-5,atol=1e-4)
                np.testing.assert_array_equal(actual[...,5],expected[...,5])
                decoder_checks.append(dict(agnostic=agnostic,input=list(shape),max_abs_error=float(np.abs(actual-expected).max()),class_indices_equal=True))
        result['selected_decoder_oracle']=decoder_checks
    (out/'export_checks.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
