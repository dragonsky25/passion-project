import pandas as pd

# sadness (0), joy (1), love (2), anger (3), fear (4), and surprise (5)

def load_data(filepath):
  data = pd.read_csv(filepath)
  data = data.rename(columns={"label": "emotion"})
  data["emotion"] = data["emotion"].replace({
  try:
    data = pd.read_csv(filepath)
    if "label" not in data.columns or "text" not in data.columns:
      raise ValueError("must contain 'label' and 'text' columns.")
    data = data.rename(columns={"label": "emotion"})
    data["emotion"] = data["emotion"].replace({
      0: "sadness", 1: "joy", 2: "love", 3: "anger", 4: "fear", 5: "surprise"
  })
  data["text"] = data["text"].str.lower()
  return data
    })
    data["text"] = data["text"].str.lower()
    return data
  except Exception as e:
    print(f"Error loading data: {e}")
    return None





