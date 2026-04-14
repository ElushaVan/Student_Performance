import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def main():
    print("Loading dataset...")
    df = pd.read_csv('Dataset/StudentPerformanceFactors.csv')
    
    # Memisahkan fitur dan target
    X = df.drop('Exam_Score', axis=1)
    y = df['Exam_Score']
    
    # Identifikasi tipe data
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Pipa pemrosesan awal (preprocessing)
    num_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    cat_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, num_cols),
            ('cat', cat_transformer, cat_cols)
        ])
    
    # Menyiapkan 3 model untuk perbandingan
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'XGBoost': XGBRegressor(n_estimators=100, random_state=42)
    }
    
    # Pembagian dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    results = {}
    best_model_name = ""
    best_r2 = -float("inf")
    best_model_pipeline = None
    
    for name, model in models.items():
        print(f"Training {name}...")
        pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                   ('model', model)])
        
        # Latih model
        pipeline.fit(X_train, y_train)
        
        # Prediksi
        y_pred = pipeline.predict(X_test)
        
        # Evaluasi
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        results[name] = {'RMSE': rmse, 'R2': r2}
        
        # Simpan model terbaik
        if r2 > best_r2:
            best_r2 = r2
            best_model_name = name
            best_model_pipeline = pipeline
            
    print("\n" + "="*40)
    print("--- Model Evaluation Results ---")
    for name, metrics in results.items():
        print(f"{name}:")
        print(f"  RMSE: {metrics['RMSE']:.2f}")
        print(f"  R-squared: {metrics['R2']:.4f}\n")
        
    print(f"** Algoritma terbaik berdasarkan R-squared: {best_model_name} **")
    print("="*40 + "\n")
    
    # Plot feature importance untuk algoritma terbaik
    model_step = best_model_pipeline.named_steps['model']
    has_importances = hasattr(model_step, 'feature_importances_')
    has_coef = hasattr(model_step, 'coef_')
    
    if has_importances or has_coef:
        print(f"Menganalisis Feature Importances dari {best_model_name}...")
        
        if has_importances:
            importances = model_step.feature_importances_
        else:
            importances = np.abs(model_step.coef_)
            
        # Dapatkan nama fitur setelah one-hot encoding
        ohe = best_model_pipeline.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot']
        cat_features = ohe.get_feature_names_out(cat_cols)
        all_features = num_cols + list(cat_features)
        
        feature_importance_df = pd.DataFrame({
            'Feature': all_features,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)
        
        top_n = min(15, len(feature_importance_df))
        top_features = feature_importance_df.head(top_n)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=top_features, palette='viridis', hue='Feature', legend=False)
        plt.title(f'Top {top_n} Faktor Paling Berpengaruh terhadap Nilai ({best_model_name})')
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300)
        print("Analisis selesai. Grafik 'feature_importance.png' berhasil disimpan di folder proyek.")

if __name__ == '__main__':
    main()
