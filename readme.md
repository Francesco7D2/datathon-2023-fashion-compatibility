# Datathon 2023 UPC MANGO

__Authors:__ Dante Chaguaceda, Aiman Himi, Adrià Rodriguez, Francesco Tedesco 

## Introduction

The main goal of this challenge is to build a model capable of creating several outfits given a single piece of clothing as input.

## Key Points
To achieve our solution, we explored several approaches. One involved filling in the blanks by grouping the data by outfits and attempting to predict the missing product given the features of this outfit and the products -1. However, this approach proved to be highly time-consuming. Therefore, we considered a different approach by rearranging the data as shown in the image below.

![image](https://github.com/Francesco7D2/datathon-2023-fashion-compatibility/assets/108528980/63223795-86dc-47b8-bd96-70068e16b22b)

- Color Definition using RGB: The RGB system was utilized to define the
colors of the products. Simultaneously, we generated three new
features for each color to create product groupings.

- Combinations of Clothing Items: We considered all possible
combinations of two clothing items. This structure involves the ID of the
first piece of clothing with its features in columns and the second piece
of clothing afterwards we did the same structure for it.


We then experimented with both neural network and Random Forest Regression for the prediction process.

## Results
By readjusting the conditions to build an outfit, we were able to establish effective rules for creating stylish outfits, as illustrated below.

![image](https://github.com/Francesco7D2/datathon-2023-fashion-compatibility/assets/108528980/b8e51f49-2203-467f-88c4-dacc48bc4ca6)


## Code
For a detailed understanding of our methodology and implementation, we welcome any feedback or suggestions.

## Files

- datathon2023_MANGO.ipynb: Jupyter Notebook with preprocessing and training code 
- model.py: Our final model for usability
- modelRFR.pkl: The best model we obtained saved.  
- presentación_Datathon.pdf: Overview of the project. 

## DEMO
Once we trained our model, we can reuse the model to predict the outfit given an initial product, as demonstrated in the image below.

[IMAGE]



