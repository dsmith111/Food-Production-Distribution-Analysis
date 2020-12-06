from sklearn.tree._tree import TREE_LEAF;
from sklearn import tree as treeDep;
import pandas as pd;
import numpy as np;


class DataProcessing():

    def __init__(self, dataFrame):
        self.dataFrame 

    def cleanDF(acceptable_array, column_name, data_frame):
        comparator = list(map(lambda element: element in acceptable_array, data_frame[column_name]))
        cleanedDF = data_frame[comparator]
        return cleanedDF

    def dropYears(data_frame, drop_years, final_year):
        drop_array = []

        for col in data_frame.columns:

            if "Y" in col:
                if final_year - int(col[1:]) >= drop_years:
                    drop_array.append(col)


        data_frame = data_frame.drop(drop_array, axis = 1)
        return data_frame

    def yieldAvg(data_frame):
        drop_array = []

        for col in data_frame.columns:

            if "Y" not in col:
                drop_array.append(col)

        data_frame_mean = data_frame.drop(drop_array, axis = 1)


        data_frame_mean["Average Yield"] = data_frame_mean.mean(axis=1)

        return(data_frame.join(data_frame_mean["Average Yield"]))


    def parallel(full_data_frame, pdata_frame, country, cereal_dictionary, column_name):


        key_df = full_data_frame[full_data_frame[column_name]==country]
        temp_dict = copy.deepcopy(cereal_dictionary)

        for cereal in cereal_dictionary.keys():

            try:
                avgYield = key_df[key_df["Item"]==cereal]["Average Yield"].iloc[0]
            except:
                avgYield = 0
            temp_dict.update({"Area":country,cereal:[avgYield]})

        temp_df = pd.DataFrame(data=temp_dict)

        pdata_frame = pdata_frame.append(temp_df, ignore_index=True)
        return pdata_frame    

    # Convert String Columns to Categorical
    def categorizeStrings(Table, targetString):

        for column in Table.columns:
            if type(Table[column][0]) == str:
                Table[column] = Table[column].astype("category")
                if column == targetString:
                    Table[column] = Table[column].cat.codes
                    continue

                Table = Table.join(pd.get_dummies(Table[column], prefix=column))
                Table = Table.drop(column, axis=1)

        return Table

    # Prune Tree
    def prune_index(inner_tree, index, threshold):
        if inner_tree.value[index].min() < threshold:
            # turn node into a leaf by "unlinking" its children
            inner_tree.children_left[index] = TREE_LEAF
            inner_tree.children_right[index] = TREE_LEAF
        # if there are children, visit them as well
        if inner_tree.children_left[index] != TREE_LEAF:
            prune_index(inner_tree, inner_tree.children_left[index], threshold)
            prune_index(inner_tree, inner_tree.children_right[index], threshold)



        
        