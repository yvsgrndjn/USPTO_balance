import pandas as pd
from .reaction_formatter import ReactionFormatterForTemplateExtraction
from .template_extractor import TemplateExtractor

def process_and_extract_templates(input_csv_path, output_csv_path):
    # Read the input DataFrame
    df = pd.read_csv(input_csv_path, index_col=0)

    # Process the reactions
    processor = ReactionFormatterForTemplateExtraction(df)
    results = processor.format_all_reactions()
    formatted_df = pd.DataFrame(results, columns=['MAPPED_SMILES', 'UNMAPPED_PRODUCT_SMILES'])

    # Extract templates for both radius 0 and radius 1
    extractor = TemplateExtractor(results)
    extracted_r0_templates = extractor.extract_r0_templates_for_all()
    extracted_r1_templates = extractor.extract_r1_templates_for_all()

    # Combine the extracted templates with the formatted DataFrame
    target = extracted_r0_templates
    formatted_df = formatted_df.assign(
        success_r0=[d['success'] for d in target],
        template_r0=[d['smarts'] for d in target],
        hash_smarts_r0=[d['hash_from_smarts'] for d in target]
    )
    
    target = extracted_r1_templates
    formatted_df = formatted_df.assign(
        success_r1=[d['success'] for d in target],
        template_r1=[d['smarts'] for d in target],
        hash_smarts_r1=[d['hash_from_smarts'] for d in target]
    )

    # Save the final DataFrame to the specified output location
    formatted_df.to_csv(output_csv_path, index=False)


