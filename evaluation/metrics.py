import sacrebleu
from sacrebleu.metrics import BLEU
from comet import download_model, load_from_checkpoint

class Evaluator:
    def __init__(self):
        self.bleu = BLEU()
        print("Loading COMET model...")
        model_path = download_model("Unbabel/wmt22-comet-da")
        self.comet_model = load_from_checkpoint(model_path)

    def eval_bleu_chrf(self, model_id, df):
        col_name = model_id.split("/")[-1]
        hyp = df[col_name].tolist()
        ref = df["target"].tolist()
        references = [[r] for r in ref]
        
        chrfpp_score = sacrebleu.corpus_chrf(hyp, [ref], word_order=2)
        bleu_score = self.bleu.corpus_score(hyp, references)

        return {
            "BLEU": bleu_score.score,
            "ChrF++": chrfpp_score.score
        }

    def eval_comet(self, model_id, df):
        col_name = model_id.split("/")[-1]
        data = [
            {
                "src": str(src).strip(),
                "mt": str(mt).strip(),
                "ref": str(ref).strip()
            }
            for src, mt, ref in zip(df["source"], df[col_name], df["target"])
        ]
        
        output = self.comet_model.predict(data, batch_size=16)
        return {'COMET': output['system_score']}