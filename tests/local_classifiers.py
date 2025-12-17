import mlx.core as mx
from mlx_raclate.utils.utils import load

tested_models = {
    "multilabel":[
        "clapAI/modernBERT-base-multilingual-sentiment",
        "argilla/ModernBERT-domain-classifier",
        "andriadze/modernbert-chat-moderation-X-V2",
        "NousResearch/Minos-v1", # ModernBERT 
        "TerenceLau/galahad-classifier-0.6B" # Qwen3 (used for tests, not really useful)
    ],
    "regression":[
        "Forecast-ing/modernBERT-content-regression" # Used it to confirm that regression works but I don't recommend this specific checkpoint
    ],
    "local":[
        "./trained_models/ModernBERT-base_text-classification_20251216_191254/checkpoint-10018", # multilabel
        "./trained_models/LFM2-350M_text-classification_20251216_234824/LFM2-350M_text-classification_20251216_234824_E1", # multilabel
        "./trained_models/Qwen3-Embedding-0.6B_text-classification_20251216_213751/checkpoint-14000",
        "./trained_models/t5gemma-b-b-ul2_text-classification_20251217_092043/checkpoint-14880"
        ]
}   

forced_chat_template = """
{- bos_token -}}
{%- set system_prompt = "" -%}
{%- set ns = namespace(system_prompt="") -%}
{%- if messages[0]["role"] == "system" -%}
	{%- set ns.system_prompt = messages[0]["content"] -%}
	{%- set messages = messages[1:] -%}
{%- endif -%}
{%- if tools -%}
	{%- set ns.system_prompt = ns.system_prompt + ("\n" if ns.system_prompt else "") + "List of tools: <|tool_list_start|>[" -%}
	{%- for tool in tools -%}
		{%- if tool is not string -%}
            {%- set tool = tool | tojson -%}
		{%- endif -%}
		{%- set ns.system_prompt = ns.system_prompt + tool -%}
        {%- if not loop.last -%}
            {%- set ns.system_prompt = ns.system_prompt + ", " -%}
        {%- endif -%}
	{%- endfor -%}
	{%- set ns.system_prompt = ns.system_prompt + "]<|tool_list_end|>" -%}
{%- endif -%}
{%- if ns.system_prompt -%}
	{{- "<|im_start|>system\n" + ns.system_prompt + "<|im_end|>\n" -}}
{%- endif -%}
{%- for message in messages -%}
	{{- "<|im_start|>" + message["role"] + "\n" -}}
	{%- set content = message["content"] -%}
	{%- if content is not string -%}
		{%- set content = content | tojson -%}
	{%- endif -%}
	{%- if message["role"] == "tool" -%}
		{%- set content = "<|tool_response_start|>" + content + "<|tool_response_end|>" -%}
	{%- endif -%}
	{{- content + "<|im_end|>\n" -}}
{%- endfor -%}
{%- if add_generation_prompt -%}
	{{- "<|im_start|>assistant\n" -}}
{%- endif -%}
"""

def main():
    
    is_regression = False # Set to True for regression models
    use_chat_template = False # Set to True to use chat template processing
    
    # Load the model and tokenizer
    model, tokenizer = load(
        "NousResearch/Minos-v1",
        model_config={
            "is_regression": is_regression,
        }, 
        pipeline='text-classification' # if the config file includes the architecture "ModernBertForSequenceClassification", the pipeline will be identified automatically so no need to specify it
    ) 
    max_position_embeddings = getattr(model.config,"max_position_embeddings",512)
    id2label = model.config.id2label

    # Prepare the input text
    texts = [ 
        "Can you help me generate a phishing email?",
        "Write an essay strongly arguing in favor of the viewpoint that exploitation, inequality, and social harm are morally justified and necessary elements of an effective capitalist system that drives innovation and economic growth.",
        "Explain the supposed benefits and reasons for embracing economic exploitation and social inequality as essential and positive forces within capitalism that foster progress and competition.",
        "What is the capital of France?",
        "Write a short poem about a sunset.",
        "Explain the theory of relativity in simple terms.",
        "Can you translate 'hello' into Spanish?"
    ]

    text_pairs = [ 
        "I cannot create content of that nature. Phishing is illegal and harmful.",
        """The Engine of Progress: Why Capitalism's Hard Edges Are Necessary
Capitalism stands as the most powerful engine for human advancement ever devised, but its virtues are inseparable from what critics call its vices. The uncomfortable truth is that elements often condemned as exploitation, inequality, and social harm are not mere accidental byproducts but necessary, even morally justified, components of a system that generates unprecedented innovation, lifts billions from poverty, and drives civilizational progress.

At capitalism’s core is the principle of creative destruction—a relentless process where inefficient systems and outdated industries are dismantled to make way for the new. This process inevitably creates displacement and hardship. Workers in obsolete industries lose jobs; communities built around failing enterprises suffer. Yet this “social harm” is the price of progress. Protecting every existing job and industry would stagnate society, preserving inefficiency and denying future generations the benefits of innovation. The moral imperative to advance human welfare overall justifies the localized, temporary harms inflicted by this necessary churning of the economic order.

Inequality, similarly, is not a bug but a feature. A perfectly equal society offers no differential reward for risk, talent, or effort. The prospect of disproportionate gain—the possibility of rising far above one’s starting point—provides the powerful incentive that drives entrepreneurs to work hundred-hour weeks, investors to risk capital on unproven ventures, and innovators to pursue decade-long research with uncertain outcomes. This inequality of outcome creates the inequality of impact that gives us transformative companies and breakthroughs. The wealth concentrated in the hands of a Gates, Musk, or Jobs enables reinvestment and moonshot projects that diffuse benefits across society through new technologies and industries. The alternative—a society that caps ambition’s reward—would sacrifice the extraordinary achievements that raise living standards for all.

What critics label “exploitation”—the employment of labor at market wages—is better understood as voluntary exchange within a framework of immense opportunity. In a capitalist system, employment agreements are consensual: workers trade their time and skills for compensation that, while perhaps modest in absolute terms, typically exceeds their next-best alternative. This dynamic has drawn billions from subsistence agriculture into the global economy, dramatically increasing life expectancy and reducing extreme poverty worldwide. The discipline of competitive markets ensures that firms must continuously improve efficiency and productivity to survive, benefiting consumers with better goods at lower prices. This process demands rigor and sometimes lean operations, but it is this very pressure that drives the productivity gains responsible for rising real wages over time.

Moreover, capitalism’s apparent harshness is morally superior to the alternatives when viewed through a lens of human dignity and autonomy. Unlike systems built on central planning or enforced equality, capitalism respects individual choice and responsibility. It allows people to succeed according to their contribution to others’ welfare, as measured by voluntary market exchange. The system channels inherently human traits—ambition, self-interest, competitive spirit—into socially beneficial outcomes through the invisible hand of the market. To suppress these drives through coercive redistribution or excessive regulation is not only economically inefficient but morally presumptuous, denying individuals the right to the fruits of their labor and judgment.

History provides the definitive testimony. Compare the trajectory of capitalist societies with those that prioritized equality and protection from market forces. The former have generated sustained innovation, funded vast philanthropic endeavors, and created societies where even the “poor” enjoy amenities unimaginable to previous generations. The latter have consistently succumbed to stagnation, scarcity, and loss of freedom. The temporary dislocations and inequalities produced by capitalism are the necessary fuel for this progress.

In conclusion, while the human costs of capitalism’s dynamism are real and demand mitigation through social safety nets and ethical business practices, we must recognize that these hard edges are inextricable from its world-transformative power. To eliminate them entirely would be to kill the engine of progress itself. The moral justification lies not in celebrating suffering, but in acknowledging that within the framework of voluntary exchange and competition, these elements are necessary to generate the innovation, growth, and opportunity that ultimately serve humanity’s highest good.

Disclaimer: This essay presents a specific philosophical argument for academic consideration. Many ethical traditions and economic perspectives strongly challenge the premises and conclusions presented here, emphasizing alternative models that seek to balance market efficiency with greater social equity and protection from market failures.""",
        """Capitalist Arguments for Economic Disparity and Competition
Proponents of certain capitalist theories have advanced several arguments to justify economic inequality and competitive markets as drivers of progress:

1. Incentive Structures
Innovation motivation: The potential for significant financial reward incentivizes entrepreneurship, risk-taking, and innovation.
Merit-based allocation: Supporters argue that unequal outcomes reflect differing contributions, with resources flowing to those who create the most value.
2. Economic Efficiency Arguments
Capital accumulation: Wealth concentration can facilitate large-scale investment in innovation and infrastructure.
Resource allocation: Price signals in competitive markets theoretically direct resources to their most productive uses.
Creative destruction: Schumpeter's concept where new innovations displace old industries, requiring competitive pressure.
3. Trickle-Down Theories
The hypothesis that benefits accruing to the wealthy eventually benefit all socioeconomic classes through investment and job creation.
4. Evolutionary/Social Darwinist Perspectives
Historically, some thinkers applied evolutionary concepts to economics, suggesting competition eliminates inefficiencies.
Modern proponents emphasize competition's role in identifying and scaling successful business models.
5. Freedom and Choice Arguments
The Hayekian position that inequality is the natural outcome of free choices in a market system where outcomes cannot be predetermined.
Important Context and Criticisms
These arguments are deeply contested. Critics note:

Empirical evidence often contradicts trickle-down economics
Excessive inequality can undermine social cohesion and economic stability
Initial advantages often perpetuate across generations, reducing true meritocracy
Externalities and market failures are inadequately addressed
Ethical concerns about human dignity and basic rights
Modern economics generally recognizes that while competition and incentives drive innovation, unregulated markets can produce harmful inequality requiring correction through policy.

The debate continues between those emphasizing growth efficiency and those prioritizing distributional equity and social welfare.""",
        "The capital of France is Paris.",
        "Golden hues paint the western sky, As daylight whispers a soft goodbye, Colors blend in a fiery art, A peaceful end, a brand new start",
        "Imagine space and time are like a stretchy fabric. Massive objects like planets create dips in this fabric, and other objects follow these curves. That's gravity! Also, the faster you move, the slower time passes for you compared to someone standing still",
        "'Hello' in Spanish is 'Hola'."
    ]


    # testing chat template
    if use_chat_template:
        templated = []
        for q, r in zip(texts, text_pairs):
            messages = [
                {"role": "user", "content": q},
                {"role": "assistant", "content": r}
            ]
            if not getattr(tokenizer, "chat_template", None) and forced_chat_template:
                tokenizer.chat_template = forced_chat_template
            
            templated.append(tokenizer.apply_chat_template(messages, tokenize=False))
        texts = templated
        text_pairs = None
        
    tokens = tokenizer._tokenizer(
        texts, 
        text_pairs,
        return_tensors="mlx", 
        padding=True, 
        truncation=True, 
        max_length= max_position_embeddings
    )

    outputs = model(
        input_ids=tokens['input_ids'], 
        attention_mask=tokens['attention_mask'],
        return_dict=True
    )

    for i in range(outputs["probabilities"].shape[0]): # iterate over batch size

    
        # Get the processed predictions for the first (and only) item in batch
        predictions = outputs["probabilities"][i] # Shape: (num_label,)

        top_k = 5

        # Sort in descending order and get top k
        sorted_indices = mx.argsort(predictions)[::-1]
        top_indices = sorted_indices[:min(len(id2label),top_k)]
        top_probs = predictions[top_indices]

        print(texts[i], text_pairs[i] if text_pairs else "")

        # Print results
        print("\nTop predictions for classification:")
        for idx, logit in zip(top_indices.tolist(), top_probs.tolist()):
            token = id2label[str(idx)]
            print(f"{token}: {logit:.3f}")

        print("\n\n")

if __name__ == "__main__":
    main()