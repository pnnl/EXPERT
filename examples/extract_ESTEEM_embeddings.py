
import argparse
import expert.embeddings import extractESTEEMembeddings


if __name__ in "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-bert', type=str, default='bert-base-uncased',
                        help='Name of bert model used for fine-tuning.')
    parser.add_argument('-model', type=str, help='path to parent_dir of pytorch_model.bin')
    parser.add_argument('-outdir', type=str, help='path to dir to save embeddings')
    parser.add_argument('-data', type=str, help='.json or jsonl file')

    arguments = parser.parse_args()
   
    extractESTEEMembeddings(arguments)
    