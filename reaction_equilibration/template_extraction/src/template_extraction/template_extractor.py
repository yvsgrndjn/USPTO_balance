from rdkit import Chem
from joblib import Parallel, delayed
from rxnutils.chem.reaction import ChemicalReaction

class TemplateExtractor:
    def __init__(self, reaction_product_pairs, n_jobs=-1):
        """Initialize with reaction-product pairs and the number of jobs for parallel processing."""
        self.reaction_product_pairs = reaction_product_pairs
        self.n_jobs = n_jobs  # Number of jobs to run in parallel (-1 uses all available processors)
        self.radius_0_results = []
        self.radius_1_results = []

    def extract_retro_template_and_hashes(self, reaction, product, radius=0):
        """Extract retro templates and hashes from a reaction and product pair."""
        try:
            rxn = ChemicalReaction(reaction)
        except ValueError:
            return False, 'rxn error', '', reaction, product, radius

        try:
            rxn.generate_reaction_template(radius=radius)
        except:
            return False, 'template error', '', reaction, product, radius

        try:
            reactant_list = rxn.retro_template.apply(product)
            return True, rxn.retro_template.smarts, rxn.retro_template.hash_from_smarts(), reaction, product, radius
        except ValueError:
            return False, rxn.retro_template.smarts, '', reaction, product, radius
        except RuntimeError as e:
            return False, f'RuntimeError: {e}', '', reaction, product, radius

    def process_single_pair(self, pair, radius):
        """Helper function to process a single reaction-product pair for a given radius."""
        reaction, product = pair
        return self.extract_retro_template_and_hashes(reaction, product, radius=radius)

    def format_results(self, results):
        """Format the results into the original dictionary format."""
        formatted_results = []
        for success, smarts, hash_from_smarts, reaction, product, radius in results:
            formatted_results.append({
                'reaction': reaction,
                'product': product,
                'radius': radius,
                'success': success,
                'smarts': smarts,
                'hash_from_smarts': hash_from_smarts,
            })
        return formatted_results

    def extract_r0_templates_for_all(self):
        """Extract templates for all reaction-product pairs at radius 0 using parallel processing."""
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(self.process_single_pair)(pair, 0) for pair in self.reaction_product_pairs
        )
        self.radius_0_results = self.format_results(results)
        return self.radius_0_results

    def extract_r1_templates_for_all(self):
        """Extract templates for all reaction-product pairs at radius 1 using parallel processing."""
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(self.process_single_pair)(pair, 1) for pair in self.reaction_product_pairs
        )
        self.radius_1_results = self.format_results(results)
        return self.radius_1_results
