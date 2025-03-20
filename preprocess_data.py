import pandas as pd

def plastic_fate_pollution():
    df = pd.read_csv('Original_Data/plastic-fate.csv')
    df = df[df['Code'] == 'OWID_WRL']
    df = df.drop(columns=['Entity', 'Code', 'Recycled', 'Incinerated', 'Landfilled'])
    df['Total Pollution'] = df['Littered'] + df['Mismanaged']
    df = df.drop(columns=['Littered', 'Mismanaged'])
    df.to_csv('Preprocessed_Data/plastic-fate.csv', index=False)


def plastic_waste_by_sector():
    df = pd.read_csv('Original_Data/plastic-waste-by-sector.csv')
    df = df.drop(columns=['Entity', 'Code'])
    df['Total Waste'] = df['Transportation'] + df['Road marking'] + df['Marine coatings'] + df['Building and construction'] + df['Packaging'] + df['Textile sector'] + df['Personal care products'] + df['Other'] + df['Industrial machinery'] + df['Consumer and institutional products'] + df['Electronics']
    df = df.drop(columns=['Transportation', 'Road marking', 'Marine coatings', 'Building and construction', 'Packaging', 'Textile sector', 'Personal care products', 'Other', 'Industrial machinery', 'Consumer and institutional products', 'Electronics'])
    df.to_csv('Preprocessed_Data/plastic-waste-by-sector.csv', index=False)

if __name__ == "__main__":
    plastic_fate_pollution()
    plastic_waste_by_sector()