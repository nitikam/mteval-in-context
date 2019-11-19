from typing import Tuple

from overrides import overrides
from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.predictors.predictor import Predictor
from allennlp.common.util import sanitize
from typing import List
from ctxteval.dataset_readers.mteval import MTReader


@Predictor.register('mteval-predictor')
class MTPredictor(Predictor):
    """
    Predictor for the :class:`~allennlp.models.bidaf.DecomposableAttention` model.
    """

    def predict(self, mtref: str, mtout: str) -> JsonDict:
        """
        Predicts whether the hypothesis is entailed by the premise text.
        Parameters
        ----------
        mtref : ``str``
            A passage representing what is assumed to be true.
        mtout : ``str``
            A sentence that may be entailed by the premise.
        Returns
        -------
        A score  .
        """
        return self.predict_json({"ref" : mtref, "sys": mtout, "srcsent": ""})['pred'][0]

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Tuple[Instance, JsonDict]:
        """
        Expects JSON that looks like ``{"premise": "...", "hypothesis": "..."}``.
        """
        ref_text = json_dict["ref"]
        mt_text = json_dict["sys"]
        src_text = json_dict["srcsent"]

        return self._dataset_reader.text_to_instance(ref_text, mt_text, src_text)

    @overrides
    def predict_batch_instance(self, instances: List[Instance]) -> List[JsonDict]:
        outputs = self._model.forward_on_instances(instances)
        outputs = sanitize(outputs)
        return [x['pred'][0] for x in outputs]
    
# python -m allennlp.service.server_simple     --archive-path esim_bert_trsys16_r3/model.tar.gz     --predictor mteval-predictor    --include-package my_library     --title "esim"     --field-name ref    --field-name sys --port 8000
