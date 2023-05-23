from jinja2 import Template
import yaml
from main import run
import multiprocessing as mp

def get_context(type='cifar',
                aggregation_methods='our_aggr',
                injective_florida=False,
                attack_methods='targeted_label_flip',
                noniid=False,
                resumed_model=True,
                mal_pcnt=0.4,
                **kwargs):
    context = {
        'type': type,
        'aggregation_methods': aggregation_methods,
        'injective_florida': injective_florida,
        'attack_methods': attack_methods,
        'noniid': noniid,
        'resumed_model': resumed_model,
        'mal_pcnt': mal_pcnt
    }
    ## add other kwargs
    for key, value in kwargs.items():
        context[key] = value
    return context

def get_param_for_context(context=None):
    # Define the dynamic values
    if context is None:
        context = get_context()

    # Load the template file
    with open('utils/jinja.yaml') as file:
        template = Template(file.read())

    # Render the template with the dynamic values
    rendered_config = template.render(context)

    # The rendered_config now contains the final YAML configuration
    # convert it into a dictionary
    params = yaml.load(rendered_config, Loader=yaml.FullLoader)

    return params


all_aggregation_methods = [ 'mean', 'geom_median','flame', 'our_aggr', 'afa', 'fltrust']

all_contexts = [get_context(aggregation_methods=aggregation_methods) for aggregation_methods in all_aggregation_methods]

all_params = [get_param_for_context(context) for context in all_contexts]

# run all params as a subprocess parallelly
def run_all_params(all_params):
    pool = mp.Pool(processes=mp.cpu_count())
    pool.map(run, all_params)
    pool.close()
    pool.join()