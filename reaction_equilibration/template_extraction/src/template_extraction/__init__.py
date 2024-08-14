__name__ = 'template_extraction'
__version__ = "0.0.1"
__author__ = "Yves Grandjean"
__email__ = "yves.grandjean@unibe.ch"

from .reaction_formatter import ReactionFormatterForTemplateExtraction
from .template_extractor import TemplateExtractor
from .pipeline import process_and_extract_templates

