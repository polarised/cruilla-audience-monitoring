import pandas as pd

df_a = pd.read_csv("/Users/joaquinarenasperez/Desktop/Repositorio-Cruilla/synthetic_cruilla_non_festival 3.csv", sep = ";")
df_b = pd.read_csv("/Users/joaquinarenasperez/Desktop/Repositorio-Cruilla/updated_allcomments.csv")[["fullText", "filter1_label"]]

df_a_sub = df_a[['fullText', 'filter1_label']]
df_b_sub = df_b[['fullText', 'filter1_label']]

combined_df = pd.concat([df_a_sub, df_b_sub], ignore_index=True)

combined_df.to_csv("combned_output.csv", index = False)


