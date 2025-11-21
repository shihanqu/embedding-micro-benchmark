import time
import statistics
import requests
from transformers import AutoTokenizer

URL = "http://192.168.198.120:1234/v1/embeddings" #default for LM Studio, but compatible with any openAI embedding endpoint if you include api key

MODELS = [
    "text-embedding-qwen3-embedding-8b",
    "text-embedding-qwen3-embedding-4b",
]


# short phrases
BASE_PHRASES_COUNT = 163
MULTIPLIER = 7
TARGET_COUNT = BASE_PHRASES_COUNT * MULTIPLIER  # 163 * 7 = 1141

# Single common words ⇒ very short, typically 1 token per phrase for most tokenizers
base_words = [
    "alpha", "beta", "gamma", "delta", "epsilon",
    "zeta", "eta", "theta", "iota", "kappa",
    "lambda", "mu", "nu", "xi", "omicron",
    "pi", "rho", "sigma", "tau", "upsilon",
    "phi", "chi", "psi", "omega",
    "red", "blue", "green", "black", "white",
    "day", "night", "sun", "moon", "star",
    "code", "data", "model", "token", "vector",
    "graph", "node", "edge", "batch", "epoch",
    "input", "output", "query", "index", "cache"
]

# Repeat the base_words list enough times and then truncate to exactly 1141 phrases
phrases = (base_words * ((TARGET_COUNT // len(base_words)) + 1))[:TARGET_COUNT]


'''
# Alternate corpus: 1/10 the phrases, each ~10x longer (paragraph-scale)
phrases = [
    "Artificial intelligence is transforming the world not just through flashy demos, but by slowly rewiring everyday workflows in software engineering, marketing, design, operations, and even traditional industries like manufacturing and logistics. Teams that once relied on manual triage of emails, tickets, and documents are quietly replacing those steps with models that categorize, summarize, and prioritize at scale. The impact is rarely a single dramatic breakthrough; it’s the compound effect of shaving minutes off thousands of repeated tasks each week. As these systems integrate more deeply into existing tools, the line between ‘using AI’ and ‘just doing your job’ starts to disappear. What used to require specialized data science expertise is now accessible to any reasonably motivated engineer through simple APIs and SDKs. The real transformation is cultural: organizations that learn to iterate quickly on AI-driven workflows gain leverage, while those that treat AI as a one-off experiment stagnate. Over the next decade, the competitive gap between these two groups will widen dramatically, and most of the difference will be explained by who actually built feedback loops around AI, rather than who merely talked about it.",

    "Machine learning is a subset of artificial intelligence, but that statement is so shallow it’s basically useless unless you understand the constraints that make machine learning valuable and dangerous at the same time. Models learn patterns from data, not truth from reality, which means they are only as good as the distributions they see and the objectives they are optimized for. When teams blindly optimize for accuracy without thinking about business goals, they ship models that technically ‘perform’ yet fail to move any real metrics. Conversely, when teams tightly couple their evaluation metrics to concrete outcomes—click-through rate, revenue, time saved, error reduction—the same mathematical machinery suddenly becomes a weaponized optimization engine. The subset that matters is not machine learning as a field, but the specific slice where your data, your loss function, and your constraints intersect. Everything else is noise and academic posturing. Until you map the math to a real-world payoff, you’re just doing expensive curve fitting.",

    "Embedding models map raw text into high dimensional vectors where geometric relationships approximate semantic relationships, but this convenience hides nontrivial engineering choices about tokenization, normalization, batching, and indexing. When you push thousands or millions of documents through an embedding pipeline, minor inefficiencies and inconsistencies compound into latency bottlenecks and retrieval failures. If you truncate text carelessly, you silently discard the context that actually matters for search relevance. If you mix embeddings from different models or from differently preprocessed text, you poison your vector space and degrade similarity results in ways that are hard to debug. Real-world systems that rely on embeddings must therefore treat the pipeline as a first-class piece of infrastructure: versioned, monitored, and tested. That means tracking which model created which vectors, validating cosine similarity distributions, and regularly checking retrieval results against hand-labeled queries. An embedding model isn’t a magic black box; it’s a fragile representational contract, and breaking that contract always shows up later as ‘weird’ search behavior and user mistrust.",

    "Latency and throughput are not abstract benchmarks; they directly determine which user experiences are even possible to build on top of an embedding service. If a single request that embeds a batch of documents takes several seconds, you can forget about tight interactive loops where users expect near-instant responses. On the other hand, if you can embed hundreds of sentences in under a second, you suddenly unlock on-the-fly personalization, dynamic reranking, and real-time analytics that weren’t practical before. Engineers who ignore performance early end up overfitting their product ideas to slow infrastructure, designing around what their system cannot do instead of what users actually want. Measuring latency distributions—p50, p90, p99—forces you to confront tail behavior, which is exactly where real users live during peak traffic and worst-case network conditions. Throughput measurements tell you whether your architecture can scale gracefully when you add more users, more documents, and more traffic. Without these numbers, you’re building blind, and your optimism about what the system ‘should be able to handle’ is just wishful thinking masquerading as planning.",

    "Tokenization is the unglamorous step that quietly determines how much you actually pay for inference, how fast your models run, and how faithfully they capture meaning from natural language. Every phrase you feed into the model is decomposed into tokens according to rules that most people never inspect. Long, compound words might be split into several subword units, while common phrases are compressed into a handful of tokens. This asymmetry means that two sentences that appear similar in character length can have dramatically different token counts, and therefore very different costs and latencies. If you never look at token distributions across your corpus, you have no idea which parts of your workload are silently inflating your bill and slowing your system. Worse, inconsistent pre-processing—like mixing raw text with aggressively cleaned text—can distort tokenization patterns in ways that hurt both performance and model behavior. Serious practitioners treat tokenization statistics as a first-class metric: they measure tokens per phrase, tokens per request, and outliers that blow up batch sizes. Everyone else just complains about ‘unexpectedly high costs’ later.",

    "Benchmarking embedding models is not just about which one feels faster in casual tests; it is about designing controlled experiments that reveal trade-offs between speed, quality, and resource usage. A naive benchmark might only measure average latency, which hides critical information about how the model behaves under load, how it scales with batch size, and how sensitive it is to input length. A serious benchmark constructs multiple corpora—short sentences, long paragraphs, mixed lengths—and runs repeated trials while collecting statistics on runtime, token counts, memory usage, and similarity performance on labeled pairs. You then compare models not in isolation but relative to constraints: maximum acceptable latency, minimum acceptable semantic quality, and budget ceilings. The winning model is rarely the one with the absolute best numbers in any single metric; it is usually the one that hits the right balance for your use case. Without this structured approach, you’re just eyeballing numbers and rationalizing whichever model you already wanted to use.",

    "Retrieval-augmented generation systems live or die on the quality of their retrieval step, and embeddings sit at the core of that process whether you acknowledge it or not. A language model, no matter how powerful, cannot compensate for garbage context fetched from a badly configured vector store. If your nearest-neighbor search returns documents that are only loosely related to the query, the generated answer will be fuzzy, generic, or outright wrong. Teams often obsess over prompt engineering and model selection while completely neglecting the embedding model, the indexing strategy, and the query transformation logic. The brutal truth is that a smaller language model with high-quality retrieval often beats a massive model with mediocre retrieval. Speed matters here too: if embedding queries and searching the index are slow, you are forced to shrink your candidate set or precompute brittle shortcuts, both of which degrade answer quality. Treating retrieval as an afterthought is just a slow way of sabotaging your own system.",

    "Scaling an embedding service from a local experiment to a production-grade system exposes all the shortcuts you took when you were just trying to ‘get something running.’ The moment you increase corpus size, concurrency, or query volume, your naive assumptions about hardware, caching, and batching collapse. CPU-only inference that was fine for a toy demo starts to choke under load, while an underutilized GPU quietly waits for you to implement proper batching and queuing. If you never measured utilization or profiled bottlenecks, you won’t know whether to invest in more hardware, better parallelism, or basic code cleanup. Worse, without clear observability—latency histograms, error rates, saturation metrics—you’ll be blind to when and why performance degrades. By the time users complain, you’re already late. Scaling is not an afterthought; it is a design constraint you either handle upfront or pay for later with outages, rushed refactors, and lost trust.",

    "Data quality in embedding workloads is a silent lever that most teams underutilize, preferring to obsess over model choice instead of fixing the text they feed into the model. Embeddings reflect whatever structure and noise is present in the underlying data, so duplicates, boilerplate, broken markup, and irrelevant fragments all pollute the vector space. When users complain that search results feel random or inconsistent, the root cause is often not the embedding model but the fact that half the indexed content is junk. Cleaning the corpus—deduplicating documents, stripping navigation text, consolidating fragmented entries, and removing obviously low-value material—has a direct, measurable impact on retrieval quality. It also reduces storage costs and speeds up indexing and querying. Yet teams still treat data cleaning as optional, then act surprised when their vector database behaves like a messy junk drawer instead of a precise retrieval engine. Blaming the model is easier than admitting that your input data is a disaster.",

    "Evaluation of embedding models is frequently done with hand-wavy ‘it looks good’ judgments instead of rigorous, labeled benchmarks that can withstand scrutiny. Engineers will run a few manual queries, skim the top results, and declare the model ‘good enough’ because they see something vaguely relevant. This is a convenient way to avoid the work of building realistic test sets: query → expected relevant documents, negative examples, and clear scoring criteria. Without such a benchmark, you cannot compare models objectively, measure regressions over time, or justify why you chose one setup over another. It also becomes impossible to have an honest conversation about trade-offs, because every disagreement turns into subjective impressions instead of data-driven discussion. If you care about search quality, you need labeled pairs, diversity in query types, and metrics like recall@k and nDCG, not just vibes. Anything less is guesswork dressed up as engineering.",

    "Monitoring an embedding-based system in production is more than checking that the service returns HTTP 200 responses; it requires tracking the health of the vector space and the consistency of the pipeline over time. New documents are constantly added, old ones updated or removed, and models occasionally upgraded. Each change subtly shifts the geometry of the embedding space. If you never measure similarity distributions, nearest-neighbor stability, or recall on a fixed test set, you’ll have no idea when your system starts drifting. That drift shows up as slowly worsening search quality, bizarre edge cases, and user reports that are easy to dismiss as anecdotal until they accumulate. Proper monitoring means logging representative queries, periodically replaying them against the index, and tracking both latency and relevance metrics. If you do none of this, you are relying on user complaints as your primary monitoring signal, which is both reactive and expensive.",

    "Token budgets impose hard limits on what you can practically embed and search, especially when dealing with long documents, logs, or multi-turn conversations. Trying to shove everything into the model without segmentation is a lazy move that guarantees wasted capacity and diluted signal. The smarter approach is to design chunking strategies that preserve semantic coherence—splitting by paragraphs, sections, or topic boundaries—rather than arbitrary fixed-length chunks. Each chunk consumes tokens, and those tokens translate directly into latency and cost. When you understand this relationship, you stop treating text as a monolithic blob and start thinking about it as structured information that can be partitioned intelligently. You also realize that some content simply isn’t worth embedding at all, and that aggressively prioritizing high-value segments is often the biggest immediate win. Ignoring token budgets is just another way of saying you’re fine wasting money and compute for no benefit.",

    "Comparing a large embedding model to a smaller one without context is meaningless; the only question that matters is whether the smaller, faster model is good enough for the specific retrieval tasks you care about. If your queries are simple and your domain is relatively clean, a compact model might perform nearly as well while being significantly cheaper and faster. If your domain is messy, multilingual, or highly technical, the larger model might capture nuances that the smaller one systematically misses. But you won’t know until you design tests that reflect your real usage patterns: ambiguous queries, domain-specific jargon, long-tail edge cases. Blindly defaulting to the biggest model you can run is not ambition; it’s laziness disguised as ‘wanting the best.’ Likewise, clinging to a small model purely for cost reasons while ignoring measurable quality gaps is just penny-wise, pound-foolish thinking.",

    "The user experience of an embedding-powered product is ultimately constrained by infrastructure choices that most end users will never see. If your APIs are slow, your UI must compensate with spinners, progress bars, and optimistic guesses about what the model will return. If your system struggles with batch processing, you’ll be forced to implement awkward, incremental updates that make the product feel jittery and unreliable. This is not just a technical inconvenience; it shapes what features your product team even dares to propose. Many ‘product decisions’ are secretly dictated by the underlying performance envelope of your ML stack. When engineers underestimate this, they overpromise capabilities that crumble at scale. When they confront it honestly, they design features that align with what the system can consistently deliver. The difference shows up in user retention, support load, and the credibility of the team.",

    "Most teams experimenting with embeddings never articulate what ‘good enough’ actually means for their use case, so they drift between models and configurations without ever converging on a stable solution. They talk vaguely about wanting ‘better search’ or ‘more relevant results’ but fail to nail down quantitative targets: specific recall levels, time-to-result constraints, or satisfaction scores tied to experiments. Without explicit goals, optimization degenerates into endless tinkering, and every new model release becomes an excuse to reset the process instead of a focused opportunity to test whether the new option moves the needle. Defining clear acceptance criteria is uncomfortable because it forces you to confront trade-offs and admit that perfection is impossible. But that discomfort is exactly where real progress starts. Until you specify what success looks like, you’re not optimizing; you’re just playing with shiny tools.",

    "The biggest blind spot in most embedding projects is the absence of a feedback loop that connects user behavior back into model and pipeline decisions. Teams ship a retrieval system once and then treat it as static infrastructure, even though user queries continually evolve as the product and audience change. Logs pile up with queries that return poor results, but no one mines them to identify systematic gaps, mislabeled documents, or index configuration issues. When you ignore this data, you leave easy wins on the table: adding domain-specific synonyms, refining chunking strategies, or creating specialized indexes for common query types. Embeddings are not a one-and-done solution; they are a component in a living system that either adapts or decays. If you refuse to close the loop with real usage data, you are effectively choosing stagnation and accepting that your search quality will slowly get worse relative to user expectations."
]
'''

'''
# Mid Corpus
phrases = [
    "Artificial intelligence is transforming the world.",
    "Machine learning is a subset of artificial intelligence.",
    "Deep learning models can achieve state of the art performance on many tasks.",
    "Neural networks are inspired by biological brains but are very different in practice.",
    "Large language models can generate coherent and context aware text.",
    "Reinforcement learning focuses on sequential decision making under uncertainty.",
    "Supervised learning relies on labeled training data.",
    "Unsupervised learning discovers hidden structure in unlabeled data.",
    "Self supervised learning uses the data itself to create training signals.",
    "Transfer learning allows models to adapt to new tasks with limited data.",
    "Fine tuning a pretrained model can dramatically improve downstream performance.",
    "Prompt engineering is a technique for steering language model behavior.",
    "Vector databases store high dimensional embeddings for similarity search.",
    "Cosine similarity is often used to compare embedding vectors.",
    "Dimensionality reduction techniques like PCA can help visualize embeddings.",
    "Clustering algorithms can group similar documents together.",
    "Tokenization converts raw text into model readable units.",
    "Subword tokenization balances vocabulary size and coverage.",
    "Embeddings map text to continuous high dimensional vectors.",
    "Semantic similarity can be approximated by vector distance.",
    "Retrieval augmented generation combines search with language models.",
    "Knowledge distillation transfers capabilities from a large model to a smaller one.",
    "Quantization reduces model size by using lower precision numbers.",
    "Pruning removes redundant parameters from neural networks.",
    "Latency and throughput are key performance metrics for inference.",
    "Batch size affects both speed and memory usage.",
    "GPU acceleration greatly speeds up deep learning workloads.",
    "CPU bound systems often struggle with very large models.",
    "Caching repeated requests can reduce average response time.",
    "Monitoring production models is essential for reliability.",
    "Model drift can degrade performance over time.",
    "Data quality strongly influences model accuracy.",
    "Bias in training data can lead to unfair outcomes.",
    "Explainability methods try to make model decisions understandable.",
    "Privacy preserving techniques aim to protect user data.",
    "Federated learning trains models across many devices without centralizing data.",
    "Differential privacy adds noise to protect individual records.",
    "A robust evaluation suite should cover many edge cases.",
    "Benchmarking different models helps choose the best tradeoffs.",
    "Smaller models are often faster but less capable.",
    "Larger models can be more accurate but require more resources.",
    "The context window length limits how much text a model can see.",
    "Embedding models focus on representation quality rather than generation.",
    "Sentence level embeddings are useful for retrieval tasks.",
    "Document embeddings allow comparison of long texts.",
    "Multilingual embeddings support many languages in one space.",
    "Cross modal embeddings connect text and images.",
    "Time series forecasting is another application of machine learning.",
    "Anomaly detection identifies unusual patterns in data.",
    "Recommendation systems personalize content for users.",
    "Search ranking models determine which results appear first.",
    "Click through rate prediction is common in online advertising.",
    "Natural language understanding is required for many applications.",
    "Intent classification maps user queries to actions.",
    "Named entity recognition identifies key concepts in text.",
    "Question answering systems retrieve or generate relevant answers.",
    "Summarization models condense long documents into short versions.",
    "Text classification models assign labels to documents.",
    "Sentiment analysis detects positive or negative opinions.",
    "Topic modeling discovers themes in collections of documents.",
    "Dialogue systems maintain context across multiple turns.",
    "Code generation models can help developers write software.",
    "Program repair models suggest fixes for buggy code.",
    "Unit tests help ensure software behaves as expected.",
    "Version control systems track changes to code bases.",
    "Continuous integration automates testing and deployment.",
    "Containers make it easier to run software in different environments.",
    "Orchestration platforms manage clusters of machines.",
    "Scaling horizontally adds more machines to handle load.",
    "Scaling vertically adds more resources to a single machine.",
    "Load balancing distributes traffic across multiple servers.",
    "Rate limiting prevents abuse of an API.",
    "Authentication verifies user identity.",
    "Authorization controls what users are allowed to do.",
    "Encryption protects data in transit and at rest.",
    "Hashing is used for efficient lookups and integrity checks.",
    "Relational databases store structured data with schemas.",
    "NoSQL databases offer flexible data models.",
    "Time series databases are optimized for temporal data.",
    "Message queues decouple producers and consumers.",
    "Event driven architectures react to changes in real time.",
    "Microservices split large systems into smaller components.",
    "Monolithic architectures keep everything in a single application.",
    "Logging is essential for debugging production systems.",
    "Metrics help quantify the health of services.",
    "Tracing shows how requests propagate through a system.",
    "Alerting notifies engineers when something goes wrong.",
    "Service level objectives define reliability targets.",
    "Incident response processes reduce downtime.",
    "Postmortems analyze failures to prevent recurrence.",
    "The sky is blue on clear days.",
    "Clouds can block sunlight and change the sky color.",
    "Rainbows appear when sunlight is refracted by water droplets.",
    "Thunderstorms can form when warm and cold air masses collide.",
    "Snow falls when atmospheric conditions are below freezing.",
    "Wind is caused by differences in air pressure.",
    "Tides are influenced by the gravitational pull of the moon.",
    "Earth orbits the sun once per year.",
    "The moon orbits Earth and causes lunar phases.",
    "Planets in the solar system follow elliptical orbits.",
    "Stars are massive spheres of hot plasma.",
    "Galaxies contain billions of stars and other matter.",
    "Black holes have gravity so strong that not even light escapes.",
    "Telescopes allow us to observe distant objects in space.",
    "Microscopes reveal structures that are invisible to the naked eye.",
    "Electricity powers most modern devices.",
    "Batteries store energy for portable electronics.",
    "Solar panels convert sunlight into electrical energy.",
    "Wind turbines generate electricity from moving air.",
    "Hydroelectric dams use flowing water to spin turbines.",
    "Mechanical systems often involve gears and levers.",
    "Friction opposes relative motion between surfaces.",
    "Gravity pulls objects toward massive bodies.",
    "Inertia causes objects to resist changes in motion.",
    "Acceleration occurs when velocity changes over time.",
    "Velocity has both magnitude and direction.",
    "Vectors represent quantities with direction and magnitude.",
    "Scalars represent quantities with only magnitude.",
    "Probability theory models uncertainty mathematically.",
    "Statistics analyzes data to draw conclusions.",
    "Correlation does not imply causation.",
    "Random variables describe uncertain numerical outcomes.",
    "Distributions characterize the likelihood of different outcomes.",
    "Bayesian methods update beliefs with new evidence.",
    "Optimization algorithms search for the best solution.",
    "Gradient descent minimizes loss functions in many models.",
    "Regularization helps prevent overfitting.",
    "Cross validation evaluates models on held out data.",
    "Hyperparameter tuning searches over configuration spaces.",
    "Grid search explores a fixed set of hyperparameters.",
    "Random search samples hyperparameters stochastically.",
    "Bayesian optimization models the performance surface.",
    "Early stopping prevents overtraining on the data.",
    "Batch normalization can stabilize neural network training.",
    "Residual connections help train very deep networks.",
    "Attention mechanisms allow models to focus on relevant inputs.",
    "Transformers rely heavily on self attention layers.",
    "Sequence to sequence models map input sequences to output sequences.",
    "Autoencoders learn compressed representations of data.",
    "Generative models attempt to create realistic samples.",
    "Diffusion models iteratively refine noisy inputs.",
    "Variational autoencoders learn probabilistic latent spaces.",
    "Generative adversarial networks pit two models against each other.",
    "Recurrent neural networks process sequences step by step.",
    "Convolutional neural networks excel at image tasks.",
    "Graph neural networks operate on graph structured data.",
    "Tabular data remains common in real world applications.",
    "ETL pipelines extract, transform, and load data.",
    "Data warehouses centralize analytics workloads.",
    "Business intelligence dashboards visualize key metrics.",
    "Key performance indicators measure progress toward goals.",
    "A longer text corpus helps benchmark embedding models.",
    "Semantically similar sentences should have nearby embeddings.",
    "Paraphrases test the nuance captured by vector representations.",
    "Contradictory statements should have lower similarity scores.",
    "Literal duplicates are a basic sanity check for embeddings.",
    "Short factual statements are useful for quick comparison.",
    "Longer sentences test the capacity of the context encoder.",
    "Technical language challenges the model vocabulary and domain knowledge.",
    "Colloquial language tests robustness to informal phrasing.",
    "Neutral sentences are good baselines for similarity measurements.",
    "Highly specific sentences stress the detail captured by embeddings.",
    "Adding noise to sentences checks stability of the representation."
]
'''

N_WARMUP = 2
N_RUNS   = 10

# ---------------- Tokenization setup ----------------

# Use a Qwen tokenizer as an approximation for the embedding models’ tokenizer
TOKENIZER_NAME = "Qwen/Qwen2.5-7B-Instruct"  # adjust if you know the exact base model
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

def count_tokens_per_phrase(texts):
    enc = tokenizer(texts, add_special_tokens=False)
    # enc["input_ids"] is list[list[int]]
    return [len(ids) for ids in enc["input_ids"]]

phrase_token_counts = count_tokens_per_phrase(phrases)
TOKENS_PER_CALL = sum(phrase_token_counts)
AVG_TOKENS_PER_PHRASE = TOKENS_PER_CALL / len(phrases)

print(f"Corpus size: {len(phrases)} phrases")
print(f"Average tokens per phrase (tokenizer={TOKENIZER_NAME}): {AVG_TOKENS_PER_PHRASE:.2f}")
print(f"Tokens per call (all phrases): {TOKENS_PER_CALL}")
print()

# ---------------- Benchmarking ----------------

def call_embedding(model, phrases):
    r = requests.post(
        URL,
        json={"model": model, "input": phrases},
        timeout=30,
    )
    r.raise_for_status()
    resp = r.json()

    if "data" not in resp or not resp["data"]:
        raise SystemExit(f"Unexpected response for model {model}: {resp}")

    # sanity check
    for i, item in enumerate(resp["data"]):
        if "embedding" not in item:
            raise SystemExit(
                f"Missing 'embedding' for model {model}, item {i}: {item}"
            )

def benchmark_model(model, phrases, n_warmup, n_runs, tokens_per_call):
    print(f"\n=== Benchmarking model: {model} ===")

    # Warmup
    for i in range(n_warmup):
        try:
            call_embedding(model, phrases)
        except Exception as e:
            raise SystemExit(f"Warmup failed for model {model}: {e}")

    durations = []

    for i in range(n_runs):
        t0 = time.perf_counter()
        try:
            call_embedding(model, phrases)
        except Exception as e:
            raise SystemExit(f"Run {i} failed for model {model}: {e}")
        t1 = time.perf_counter()

        dt = t1 - t0
        durations.append(dt)
        print(f"  Run {i+1:02d}: {dt:.4f} s")

    total_time = sum(durations)
    total_tokens = tokens_per_call * n_runs

    avg = statistics.mean(durations)
    p50 = statistics.median(durations)
    p90 = statistics.quantiles(durations, n=10)[8] if len(durations) >= 10 else None
    mn = min(durations)
    mx = max(durations)

    tokens_per_second = total_tokens / total_time
    phrases_per_second = (len(phrases) * n_runs) / total_time

    result = {
        "model": model,
        "runs": n_runs,
        "min_s": mn,
        "max_s": mx,
        "avg_s": avg,
        "p50_s": p50,
        "p90_s": p90,
        "phrases_per_call": len(phrases),
        "tokens_per_call": tokens_per_call,
        "tokens_per_second": tokens_per_second,
        "phrases_per_second": phrases_per_second,
        "avg_tokens_per_phrase": AVG_TOKENS_PER_PHRASE,
    }

    return result


def main():
    all_results = []

    for model in MODELS:
        res = benchmark_model(model, phrases, N_WARMUP, N_RUNS, TOKENS_PER_CALL)
        all_results.append(res)

    print("\n=== Summary ===")
    for r in all_results:
        print(f"\nModel: {r['model']}")
        print(f"  Calls:               {r['runs']}")
        print(f"  Phrases per call:    {r['phrases_per_call']}")
        print(f"  Avg tokens/phrase:   {r['avg_tokens_per_phrase']:.2f}")
        print(f"  Tokens per call:     {r['tokens_per_call']}")
        print(f"  Min latency:         {r['min_s']:.4f} s")
        print(f"  P50 latency:         {r['p50_s']:.4f} s")
        if r['p90_s'] is not None:
            print(f"  P90 latency:         {r['p90_s']:.4f} s")
        print(f"  Max latency:         {r['max_s']:.4f} s")
        print(f"  Avg latency:         {r['avg_s']:.4f} s")
        print(f"  Tokens / second:     {r['tokens_per_second']:.1f}")
        print(f"  Phrases / second:    {r['phrases_per_second']:.2f}")


if __name__ == "__main__":
    main()
