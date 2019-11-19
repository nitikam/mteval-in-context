from typing import Dict
import json
import logging

from allennlp.data import Token
from overrides import overrides
from pytorch_pretrained_bert import BertTokenizer
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, TextField, MetadataField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Tokenizer, WordTokenizer
from ctxteval.dataset_readers.numeric_field import NumericField
logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

from numpy import isnan

SEP_TOKEN: str = "[SEP]"
START_TOKEN: str = "[CLS]"

@DatasetReader.register("mteval")
class MTReader(DatasetReader):
    """
    Reads a file from the Stanford Natural Language Inference (SNLI) dataset.  This data is
    formatted as jsonl, one json-formatted instance per line.  The keys in the data are
    "gold_label", "sentence1", and "sentence2".  We convert these keys into fields named "label",
    "premise" and "hypothesis", along with a metadata field containing the tokenized strings of the
    premise and hypothesis.

    Parameters
    ----------
    tokenizer : ``Tokenizer``, optional (default=``WordTokenizer()``)
        We use this ``Tokenizer`` for both the premise and the hypothesis.  See :class:`Tokenizer`.
    token_indexers : ``Dict[str, TokenIndexer]``, optional (default=``{"tokens": SingleIdTokenIndexer()}``)
        We similarly use this for both the premise and the hypothesis.  See :class:`TokenIndexer`.
    """

    def __init__(self,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 maxlen: int = 100,
                 bert_name: str = None,
                 bert_do_lowercase: bool = None,
                 qesetting: bool = False,
                 inp_type: str = "metric",
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        self._tokenizer = tokenizer or WordTokenizer()
        self.bert_name = bert_name
        if bert_name and not bert_do_lowercase:
            if 'uncased' in bert_name:
                bert_do_lowercase = True
            else:
                bert_do_lowercase = False
        if bert_name:
            self._tokenizer = BertTokenizer.from_pretrained(bert_name, do_lower_case = bert_do_lowercase)
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        self._maxlen = maxlen
        self._qesetting = qesetting
        valid_inp_types = ["metric",  "qe" ]
        assert  inp_type in valid_inp_types
        self._inp_type = inp_type

    @overrides
    def _read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        with open(file_path, 'r') as snli_file:
            logger.info("Reading MT instances from jsonl dataset at: %s", file_path)
            for line in snli_file:
                example = json.loads(line)
                score = example.get( "score" , 0)
                if isnan(score):
                        continue


                src = example.get('srcsent', "")
                ref = example["ref"]
                mt = example["sys"]


                yield self.text_to_instance(ref, mt, src= src,
                                                 score= score)


    @overrides
    def text_to_instance(self,  # type: ignore
                         ref: str,
                         mt: str,
                         src: str = "",
                         score: str = None) -> Instance:
        # pylint: disable=arguments-differ
        fields: Dict[str, Field] = {}

        ref_tokens =   self._tokenizer.tokenize(ref)[:self._maxlen]
        mt_tokens =   self._tokenizer.tokenize(mt)[:self._maxlen]
        src_tokens =  self._tokenizer.tokenize(src)[:self._maxlen]

        if self.bert_name:
            ref_tokens = [Token(t) for t in ref_tokens]
            mt_tokens = [Token(t) for t in mt_tokens]
            src_tokens = [Token(t) for t in src_tokens]


        if self._inp_type == 'metric':
            fields['mt'] = TextField(mt_tokens, self._token_indexers)
            if self._qesetting: #this is just for backwards compatibility
                fields['ref'] = TextField(src_tokens, self._token_indexers)
            else:
                fields['ref'] = TextField(ref_tokens, self._token_indexers)

        elif self._inp_type == 'qe':
            fields['ref'] = TextField(src_tokens, self._token_indexers)
            fields['mt'] = TextField(mt_tokens, self._token_indexers)

        if score is not None:
            fields['score'] = NumericField(score)
        # metadata = {"ref_tokens": [x.text for x in ref_tokens],
        #             "mt_tokens": [x.text for x in mt_tokens]}
        # fields["metadata"] = MetadataField(metadata)
        return Instance(fields)
