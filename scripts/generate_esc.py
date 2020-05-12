import argparse
import os
from tqdm import tqdm
import torch
from fairseq.models.esc_model import ESCModel

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default=None, help="Sentence compression input path.")
    parser.add_argument("--output_path", type=str, default=None, help="Sentence compression input path.")
    parser.add_argument("--esc_model_path", type=str, default=None, help="Sentence compression model path.")
    parser.add_argument("--esc_batch_size", type=int, default=64, help="Sentence compression model beam size.")
    parser.add_argument("--esc_beam", type=int, default=4, help="Sentence compression model beam size.")
    parser.add_argument("--esc_lenpen", type=float, default=2.0, help="Sentence compression model length penalty.")
    parser.add_argument("--esc_max_len_a", type=float, default=0.0, help="Sentence compression model max_len_a.")
    parser.add_argument("--esc_max_len_b", type=int, default=140, help="Sentence compression model max_len_b.")
    parser.add_argument("--esc_min_len", type=int, default=40, help="Sentence compression model min_len.")
    parser.add_argument("--esc_no_repeat_ngram_size", type=int, default=3, help="Sentence compression model no_repeat_ngram_size.")
    args = parser.parse_args()

    esc_model = ESCModel.from_pretrained(args.esc_model_path, checkpoint_file='model.pt')
    esc_model.cuda()
    esc_model.eval()
    esc_model.half()
    
    with open(args.input_path, 'r') as fin:
        data = fin.readlines()

    bsz = args.esc_batch_size
    esc_inputs = []
    masks = []

    with open(args.output_path, 'w') as fout:
        for line_idx, line in tqdm(enumerate(data)):
            assert len(line.strip()) > 0

            if len(esc_inputs) == bsz:
                esc_outputs_batch = []
                clean_esc_inputs = [esc_inputs[idx] for idx,mask in enumerate(masks) if mask==1]
                if len(clean_esc_inputs) > 0:
                    with torch.no_grad():
                        hypotheses_batch = esc_model.sample(clean_esc_inputs, beam=args.esc_beam, lenpen=args.esc_lenpen, max_len_a=args.esc_max_len_a, max_len_b=args.esc_max_len_b, min_len=args.esc_min_len, no_repeat_ngram_size=args.esc_no_repeat_ngram_size)
                    jdx = 0
                    for idx,mask in enumerate(masks):
                        if mask == 0:
                            esc_outputs_batch.append(esc_inputs[idx])
                        else:
                            esc_outputs_batch.append(hypotheses_batch[jdx])
                            jdx += 1
                    assert jdx == len(hypotheses_batch)
                else:
                    esc_outputs_batch = esc_inputs
                assert len(esc_outputs_batch) == len(esc_inputs)
                for hypo in esc_outputs_batch:
                    fout.write(hypo.replace('\n', ' ')+'\n')
                esc_inputs = []
                masks = []
            
            if len(line.strip().split()) < args.esc_min_len:
                masks.append(0)
            else:
                masks.append(1)
            esc_inputs.append(line)
        
        # 
        if len(esc_inputs) > 0:
            esc_outputs_batch = []
            clean_esc_inputs = [esc_inputs[idx] for idx,mask in enumerate(masks) if mask==1]
            if len(clean_esc_inputs) > 0:
                with torch.no_grad():
                    hypotheses_batch = esc_model.sample(clean_esc_inputs, beam=args.esc_beam, lenpen=args.esc_lenpen, max_len_a=args.esc_max_len_a, max_len_b=args.esc_max_len_b, min_len=args.esc_min_len, no_repeat_ngram_size=args.esc_no_repeat_ngram_size)
                jdx = 0
                for idx,mask in enumerate(masks):
                    if mask == 0:
                        esc_outputs_batch.append(esc_inputs[idx])
                    else:
                        esc_outputs_batch.append(hypotheses_batch[jdx])
                        jdx += 1
                assert jdx == len(hypotheses_batch)
            else:
                esc_outputs_batch = esc_inputs
            assert len(esc_outputs_batch) == len(esc_inputs)
            for hypo in esc_outputs_batch:
                fout.write(hypo.replace('\n', ' ')+'\n')

            

    
