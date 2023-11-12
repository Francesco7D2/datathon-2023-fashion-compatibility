"""
File: predictor.py
Authors: Dante Chaguaceda, Aiman Himi, Adrià Rodriguez, Francesco Tedesco
Created: November 12, 2023
Description: This file contains the prediction process of the MANGO challenge in UPC datathon 2023
"""


import numpy as np
import pandas as pd
import category_encoders as ce

# Define the color data
colors_rgb = {'OFFWHITE': np.array([250, 249, 246]),
              'TEJANO OSCURO': np.array([52, 63, 81]),
              'ROSA PASTEL': np.array([253,202,225]),
              'MOSTAZA': np.array([246,167,0]),
              'ROJO': np.array([255, 0, 0]),
              'GRIS MEDIO VIGORE': np.array([74,80,83]),
              'CARAMELO': np.array([174,105,56]),
              'KHAKI': np.array([240, 230, 140]),
              'GRIS OSCURO VIGORE': np.array([51, 47, 44]),
              'CRUDO': np.array([194, 178, 128]),
              'FRESA': np.array([213, 48, 50]),
              'BEIGE': np.array([245, 245, 220]),
              'CHOCOLATE': np.array([122, 46, 17]),
              'NEGRO': np.array([0, 0, 0]),
              'ROSA PALO': np.array([255, 192, 203]),
              'TEJANO MEDIO': np.array([87, 110, 138]),
              'AMARILLO': np.array([255, 255, 0]),
              'BLANCO': np.array([255, 255, 255]),
              'ARENA': np.array([194, 178, 128]),
              'MORADO': np.array([128, 0, 128]),
              'MARRON': np.array([139, 69, 19]),
              'AZUL': np.array([0, 0, 255]),
              'AGUA': np.array([0, 255, 255]),
              'NAVY': np.array([0, 0, 128]),
              'TEJANO GRIS OSCURO': np.array([70, 70, 70]),
              'CAMEL': np.array([193, 154, 107]),
              'VERDE': np.array([0, 128, 0]),
              'ROSA': np.array([255, 182, 193]),
              'FUCSIA': np.array([255, 0, 255]),
              'GRIS CLARO VIGORE': np.array([200, 200, 200]),
              'CUERO': np.array([139, 69, 19]),
              'VISON': np.array([138, 110, 93]),
              'TEJANO CLARO': np.array([173, 216, 230]),
              'PLATA': np.array([192, 192, 192]),
              'GRIS': np.array([128, 128, 128]),
              'NUDE': np.array([222, 184, 135]),
              'ORO': np.array([255, 215, 0]),
              'TURQUESA': np.array([64, 224, 208]),
              'TEJANO NEGRO': np.array([0, 0, 0]),
              'NARANJA': np.array([255, 165, 0]),
              'DIRTY': np.array([131, 111, 255]),
              'HIELO': np.array([176, 224, 230]),
              'CELESTE': np.array([173, 216, 230]),
              'BURDEOS': np.array([128, 0, 32]),
              'AMARILLO FLUOR': np.array([255, 255, 85]),
              'PIEDRA': np.array([162, 128, 84]),
              'MALVA': np.array([221, 160, 221]),
              'ROSA LIGHT': np.array([255, 182, 193]),
              'MARINO': np.array([0, 0, 128]),
              'MANDARINA': np.array([255, 165, 0]),
              'TERRACOTA': np.array([204, 78, 92]),
              'CALDERO': np.array([127, 127, 127]),
              'GRANATE': np.array([128, 0, 0]),
              'TOPO': np.array([117, 109, 102]),
              'VERDE PASTEL': np.array([152, 251, 152]),
              'AZUL NOCHE': np.array([25, 25, 112]),
              'MISTERIO': np.array([50, 50, 50]),
              'LIMA': np.array([0, 255, 0]),
              'COFFEE': np.array([111, 78, 55]),
              'PERLA': np.array([234, 234, 234]),
              'ESMERALDA': np.array([0, 201, 87]),
              'OCRE': np.array([204, 119, 34]),
              'VIOLETA': np.array([238, 130, 238]),
              'VINO': np.array([128, 0, 0]),
              'MARFIL': np.array([255, 255, 240]),
              'BOTELLA': np.array([0, 128, 0]),
              'ANTRACITA': np.array([45, 45, 45]),
              'PEACH': np.array([255, 218, 185]),
              'VAINILLA': np.array([245, 245, 220]),
              'TABACO': np.array([165, 42, 42]),
              'ELECTRICO': np.array([125, 249, 255]),
              'AMARILLO PASTEL': np.array([255, 255, 182]),
              'PORCELANA': np.array([240, 240, 240]),
              'TAUPE': np.array([72, 60, 50]),
              'COBRE': np.array([184, 115, 51]),
              'CANELA': np.array([210, 105, 30]),
              'PETROLEO': np.array([0, 128, 128]),
              'CORAL': np.array([255, 127, 80]),
              'LILA': np.array([200, 162, 200]),
              'COGNAC': np.array([156, 82, 45]),
              'OLIVA': np.array([128, 128, 0]),
              'MENTA': np.array([0, 255, 128]),
              'TEJANO GRIS CLARO': np.array([192, 192, 192]),
              'TEJANO GRIS': np.array([128, 128, 128]),
              'DIRTY OSCURO': np.array([75, 0, 130]),
              'BLEACH': np.array([255, 255, 224]),
              'NARANJA PASTEL': np.array([255, 218, 185]),
              'CAZA': np.array([139, 69, 19]),
              'BLOOD': np.array([102, 0, 0]),
              'CEREZA': np.array([222, 49, 99]),
              'CENIZA': np.array([192, 192, 192]),
              'CURRY': np.array([128, 128, 0]),
              'BLUEBLACK': np.array([0, 0, 139]),
              'SALMON': np.array([250, 128, 114]),
              'CHICLE': np.array([255, 182, 193]),
              'TEJANO SOFT': np.array([173, 216, 230]),
              'GUNMETAL': np.array([42, 42, 42]),
              'MANZANA': np.array([0, 128, 0]),
              'BILLAR': np.array([0, 128, 128]),
              'MUSGO': np.array([0, 128, 0]),
              'TINTA': np.array([0, 0, 128]),
              'INDIGO': np.array([75, 0, 130]),
              'ROSA FLUOR': np.array([255, 0, 255]),
              'DIRTY CLARO': np.array([171, 171, 171]),
              'CIRUELA': np.array([221, 160, 221]),
              'BERMELLON': np.array([255, 69, 0]),
              'GERANIO': np.array([255, 69, 0]),
              'PIMENTON': np.array([176, 23, 31]),
              'PRUSIA': np.array([0, 49, 83]),
              'ASFALTO': np.array([24, 24, 24]),
  }
# Normalize the data
for k, v in colors_rgb.items():
    colors_rgb[k] = v/255


class Predictor():
    """
    Predictor class for building outfits given a single item of clothing.

    Parameters:
    - product_dataset_path (str): Path to the product dataset file.
    - classifier: Classifier model used for predictions.

    Attributes:
    - raw_dataset (pd.DataFrame): Raw product dataset.
    - dataset (pd.DataFrame): Processed dataset.
    - classifier: Classifier model for predictions.
    - not_with_it_cat (dict): Categories not to be paired together.
    - not_with_it_type (dict): Types not to be paired together.

    Methods:
    - __init__: Initialize the Predictor instance.
    - get_most_similars(row): Get indices of most similar items given a row.
    - get_img_path_from_prediciton(outfit): Get image paths from a prediction outfit.
    - get_predictions(row, prediction, n, n_outfits): Generate multiple outfits based on predictions.
    - predict(id, n_images, n_outfits): Make predictions for outfit images.

    """
    def __init__(self, product_dataset_path: str, classifier):
        self.raw_dataset = pd.read_csv(product_dataset_path)
        self.raw_dataset = self.raw_dataset[self.raw_dataset['des_sex'] == "Female"]
        self.raw_dataset = self.raw_dataset[self.raw_dataset['des_age'] == "Adult"]
        self.dataset = self.raw_dataset.copy()
        
        # Change colors
        self.dataset[['R', 'G', 'B']] = pd.DataFrame(self.dataset['des_color_specification_esp'].map(colors_rgb).tolist(), index=self.dataset.index)
        # Delete unuseful columns
        self.dataset.drop(columns=["des_sex", "des_age",  'des_color_specification_esp', "des_agrup_color_eng", "cod_color_code",  "des_line",  "des_product_type", "des_product_family", "des_product_aggregated_family", "des_product_category",  "des_filename"], axis=1, inplace=True)
        # One-Hot encoding
        ce_OHE = ce.OneHotEncoder(cols=['des_fabric']) 
        self.dataset = ce_OHE.fit_transform(self.dataset)
        
        
        self.classifier = classifier


        self.not_with_it_cat = {"Bottoms": set(["Bottoms"]), "Dresses, jumpsuits and Complete set": set(["Tops", "Dresses, jumpsuits and Complete set"]),
                                "Tops": set(["Tops"]), "Outerwear": set(["Outerwear"])}
        self.not_with_it_type = {"Dress": set(["Skirt", "Shorts", "T-Shirt"]), "Scarf": set(["Scarf"]), "Jacket": set(["Jacket"]),
                                 "Umbrella": set(["Umbrella"])}
        
        
    def get_most_similars(self, row):
        # Convert the row to a DataFrame with a single row
        row_df = pd.DataFrame(row).T
        # Merge the row with the dataset using 'cross' method
        internal_ds_1 = pd.merge(row_df, self.dataset, how='cross')
        internal_ds_2 = pd.merge(self.dataset, row_df, how='cross')
        
        # internal_ds.reset_index(inplace=True)
        internal_ds_1.drop(columns=["cod_modelo_color_x", "cod_modelo_color_y"], axis=1, inplace=True)
        internal_ds_2.drop(columns=["cod_modelo_color_x", "cod_modelo_color_y"], axis=1, inplace=True)
        i = internal_ds_1.to_numpy(dtype="float64")
        j = internal_ds_2.to_numpy(dtype="float64")
        y_pred = np.maximum(self.classifier.predict(i), self.classifier.predict(j))
        return list(reversed(np.argsort(y_pred).tolist()))
    
    
    def get_img_path_from_prediciton(self, outfit):
        paths = []
        for el in outfit:
            paths.append(el["des_filename"])
        return paths
    
    
    def get_predictions(self, row, prediction, n: int, n_outfits):
        outfits = []
        if n_outfits > 0:
            i, new_outfit = self.get_predictions(row, prediction, n, n_outfits-1)
            outfits.extend(new_outfit)
        else:
            i = 0
        outfit = [row]
        i_prev = -1
        new_i = -1
        steps_to_save = n//2
        for _ in range(n):
            while i != i_prev:
                i_prev = i
                pred = self.raw_dataset.iloc[prediction[i], :]
                if row["cod_modelo_color"] == pred["cod_modelo_color"]: # Mateixa prenda
                    i += 1
                    continue
                
                r_cat = [r["des_product_category"] for r in outfit]
                p_cat = pred["des_product_category"]
                r_type = [r["des_product_type"] for r in outfit]
                p_type = pred["des_product_type"]
                for k, v in self.not_with_it_cat.items():
                    if (k in r_cat and p_cat in v) or (len(v.intersection(r_cat))!=0 and p_cat == k):
                        i += 1
                        continue
                
                for k, v in self.not_with_it_type.items():
                    if (k in r_type and p_type in v) or (len(v.intersection(r_type))!=0 and p_type == k):
                        i += 1
                        continue
                    
            outfit.append(pred)
            i += 1
            steps_to_save -= 1
            if steps_to_save < 0:
                new_i = i
                steps_to_save = 100000
        outfits.append(outfit)
        return new_i, outfits
    
    
    def predict(self, id: str|int, n_images: int, n_outfits: int):
        index = id if type(id) == int else self.raw_dataset.index[self.raw_dataset["cod_modelo_color"] == id].tolist()[0]
        index_sorted = self.get_most_similars(self.dataset.loc[index, :])
        _, outfits = self.get_predictions(self.raw_dataset.loc[index, :], index_sorted, n_images-1, n_outfits)
        return [self.get_img_path_from_prediciton(outfit) for outfit in outfits]
        
        

        
        
