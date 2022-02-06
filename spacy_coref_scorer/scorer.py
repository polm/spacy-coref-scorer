from .coval import Evaluator, get_cluster_info, b_cubed, muc, ceafe
import spacy

DEFAULT_CLUSTER_PREFIX = "coref_clusters"

def doc2clusters(doc: Doc, prefix=DEFAULT_CLUSTER_PREFIX) -> MentionClusters:
    """Given a doc, give the mention clusters."""
    out = []
    for name, val in doc.spans.items():
        if not name.startswith(prefix):
            continue

        cluster = []
        for mention in val:
            cluster.append((mention.start, mention.end))
        out.append(cluster)
    return out

@spacy.registry.scorers("coref.v1")
def score(self, examples, **kwargs):
    """Score a batch of examples."""

    # NOTE traditionally coref (conll2003) uses the average of b_cubed, muc,
    # and ceaf. we need to handle the average ourselves.
    # Here we'll create a dictionary with scores for each function and average
    # them afterwards.
    scores = []
    prefix = kwargs.get("prefix", DEFAULT_CLUSTER_PREFIX)
    for metric in (b_cubed, muc, ceafe):
        evaluator = Evaluator(metric)

        for ex in examples:
            p_clusters = doc2clusters(ex.predicted, prefix)
            g_clusters = doc2clusters(ex.reference, prefix)

            cluster_info = get_cluster_info(p_clusters, g_clusters)

            evaluator.update(cluster_info)

        score = {
            "coref_f": evaluator.get_f1(),
            "coref_p": evaluator.get_precision(),
            "coref_r": evaluator.get_recall(),
        }
        scores.append(score)

    out = {}
    for field in ("f", "p", "r"):
        fname = f"coref_{field}"
        out[fname] = mean([ss[fname] for ss in scores])
    return out
