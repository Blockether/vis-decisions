"""Model identity shared by the optional GLiNER trainer and lightweight publisher."""

ARCHITECTURES = {
    "gliner2.5-base": "boundary",
    "gliner2.5-small": "boundary",
    "gliner2.5-multi": "boundary",
    "gliner2.5-decide": "span",
    "gliner2.5-multi-decide": "boundary",
    "gliner2.5-decide-1b": "span",
}

ENCODERS = {
    "gliner2.5-base": "deberta-v2",
    "gliner2.5-small": "deberta-v2",
    "gliner2.5-multi": "deberta-v2",
    "gliner2.5-decide": "deberta-v2",
    "gliner2.5-multi-decide": "deberta-v2",
    "gliner2.5-decide-1b": "modernbert",
}
