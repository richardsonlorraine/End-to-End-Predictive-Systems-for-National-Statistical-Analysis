import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
def plot_feature_importance(model, feature_names: list, top_n: int = 10):
    importances = model.feature_importances_
    feat_imp = pd.DataFrame({'feature': feature_names, 'importance': importances})
    feat_imp = feat_imp.sort_values(by='importance', ascending=False).head(top_n)
    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=feat_imp, palette='viridis', hue='feature', legend=False)
    plt.title(f"Top {top_n} Features Driving Net Migration/Population Growth")
    plt.xlabel("Feature Importance Score")
    plt.ylabel("Features")
    plt.tight_layout()
    plt.savefig("data/processed/feature_importance.png")
    plt.show()