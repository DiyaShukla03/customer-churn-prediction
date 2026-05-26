import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
import joblib
import warnings
import os
warnings.filterwarnings('ignore')

print("=" * 60)
print("    CUSTOMER CHURN PREDICTION SYSTEM")
print("=" * 60)

print("\n[1/6] Creating dataset...")
np.random.seed(42)
n = 10000

tenure = np.random.randint(0, 72, n)
monthly = np.random.uniform(20, 120, n)
support = np.random.randint(0, 10, n)
late = np.random.randint(0, 6, n)
contract = np.random.choice(['Month-to-Month', 'One Year', 'Two Year'], n, p=[0.5, 0.3, 0.2])
payment = np.random.choice(['Electronic Check', 'Mailed Check', 'Bank Transfer', 'Credit Card'], n)
age = np.random.randint(18, 70, n)
products = np.random.randint(1, 5, n)
internet = np.random.choice([0, 1], n, p=[0.2, 0.8])
phone = np.random.choice([0, 1], n, p=[0.1, 0.9])
senior = np.random.choice([0, 1], n, p=[0.8, 0.2])
partner = np.random.choice([0, 1], n)
dependents = np.random.choice([0, 1], n, p=[0.7, 0.3])
paperless = np.random.choice([0, 1], n)
total = monthly * (tenure + 1) + np.random.uniform(-50, 50, n)

score = np.zeros(n)
score += np.where(contract == 'Month-to-Month', 4.0, 0)
score += np.where(contract == 'One Year', 1.0, 0)
score += np.where(contract == 'Two Year', -3.0, 0)
score += np.where(tenure < 6, 3.0, 0)
score += np.where(tenure < 12, 1.5, 0)
score += np.where(tenure > 36, -3.0, 0)
score += np.where(tenure > 48, -2.0, 0)
score += support * 0.5
score += late * 0.8
score += np.where(monthly > 85, 1.5, 0)
score += np.where(monthly < 35, -1.0, 0)
score += np.where(senior == 1, 1.0, 0)
score += np.where(partner == 1, -1.0, 0)
score += np.where(dependents == 1, -1.0, 0)
score += np.where(products >= 3, -1.5, 0)
score += np.where(payment == 'Electronic Check', 1.0, 0)
score += np.where(payment == 'Credit Card', -0.5, 0)

prob = 1 / (1 + np.exp(-score))
churn = (np.random.random(n) < prob).astype(int)

df = pd.DataFrame({
    'age': age, 'tenure': tenure, 'monthly_charges': monthly,
    'total_charges': total, 'num_products': products,
    'has_internet': internet, 'has_phone': phone, 'is_senior': senior,
    'has_partner': partner, 'has_dependents': dependents,
    'paperless_billing': paperless, 'support_calls': support,
    'late_payments': late, 'contract_type': contract,
    'payment_method': payment, 'churn': churn
})

print(f"   Done: {n} rows, Churn rate: {churn.mean():.1%}")

print("\n[2/6] Performing EDA...")
os.makedirs('static/plots', exist_ok=True)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Customer Churn - EDA', fontsize=16, fontweight='bold')

cc = df['churn'].value_counts()
axes[0,0].pie(cc.values, labels=['No Churn','Churn'], autopct='%1.1f%%', colors=['#2ecc71','#e74c3c'])
axes[0,0].set_title('Churn Distribution')

axes[0,1].hist(df[df['churn']==0]['tenure'], alpha=0.7, label='Stay', color='#2ecc71', bins=20)
axes[0,1].hist(df[df['churn']==1]['tenure'], alpha=0.7, label='Churn', color='#e74c3c', bins=20)
axes[0,1].set_title('Tenure by Churn')
axes[0,1].legend()

ct = df.groupby('contract_type')['churn'].mean().sort_values(ascending=False)
bars = axes[0,2].bar(ct.index, ct.values, color=['#e74c3c','#f39c12','#2ecc71'])
axes[0,2].set_title('Churn Rate by Contract')
axes[0,2].tick_params(axis='x', rotation=15)
for b, v in zip(bars, ct.values):
    axes[0,2].text(b.get_x()+b.get_width()/2, b.get_height()+0.01, f'{v:.1%}', ha='center', fontweight='bold')

axes[1,0].boxplot([df[df['churn']==0]['monthly_charges'], df[df['churn']==1]['monthly_charges']],
                  labels=['Stay','Churn'], patch_artist=True)
axes[1,0].set_title('Monthly Charges vs Churn')

sc = df.groupby('support_calls')['churn'].mean()
axes[1,1].bar(sc.index, sc.values, color='#e74c3c')
axes[1,1].set_title('Churn by Support Calls')

lc = df.groupby('late_payments')['churn'].mean()
axes[1,2].bar(lc.index, lc.values, color='#e67e22')
axes[1,2].set_title('Churn by Late Payments')

plt.tight_layout()
plt.savefig('static/plots/eda_analysis.png', dpi=150, bbox_inches='tight')
plt.close()

plt.figure(figsize=(10, 8))
nc = df.select_dtypes(include=[np.number]).columns.tolist()
corr = df[nc].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', center=0)
plt.title('Correlation Heatmap', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('static/plots/correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("   Done: plots saved")

print("\n[3/6] Feature Engineering...")
dml = df.copy()
lec = LabelEncoder()
lep = LabelEncoder()
dml['contract_encoded'] = lec.fit_transform(dml['contract_type'])
dml['payment_encoded'] = lep.fit_transform(dml['payment_method'])
dml.drop(['contract_type','payment_method'], axis=1, inplace=True)
dml['avg_spend'] = dml['total_charges'] / (dml['tenure'] + 1)
dml['charge_ratio'] = dml['monthly_charges'] / (dml['tenure'] + 1)
dml['high_value'] = (dml['monthly_charges'] > 80).astype(int)
dml['long_tenure'] = (dml['tenure'] > 24).astype(int)
dml['new_customer'] = (dml['tenure'] < 6).astype(int)
dml['risk_score'] = dml['support_calls']*0.35 + dml['late_payments']*0.45 + (1-dml['long_tenure'])*0.2
dml['loyalty'] = dml['tenure']*0.4 + dml['num_products']*0.3 + dml['has_partner']*0.15 + dml['has_dependents']*0.15
print(f"   Done: {dml.shape[1]-1} features")

os.makedirs('models', exist_ok=True)
joblib.dump(lec, 'models/encoder_contract.pkl')
joblib.dump(lep, 'models/encoder_payment.pkl')

print("\n[4/6] Preparing data...")
X = dml.drop('churn', axis=1)
y = dml['churn']
fnames = list(X.columns)
joblib.dump(fnames, 'models/feature_names.pkl')

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
sc = StandardScaler()
Xtr_s = sc.fit_transform(Xtr)
Xte_s = sc.transform(Xte)
joblib.dump(sc, 'models/scaler.pkl')
print(f"   Train: {Xtr.shape[0]}, Test: {Xte.shape[0]}, Features: {Xtr.shape[1]}")

print("\n[5/6] Training Models...")
lr = LogisticRegression(max_iter=2000, C=1.0, random_state=42)
lr.fit(Xtr_s, ytr)
lrp = lr.predict(Xte_s)
lrpr = lr.predict_proba(Xte_s)[:,1]
lra = accuracy_score(yte, lrp)
lrauc = roc_auc_score(yte, lrpr)
print(f"   LR  Accuracy: {lra*100:.2f}%  AUC: {lrauc:.4f}")

rf = RandomForestClassifier(n_estimators=300, max_depth=20, min_samples_split=2, min_samples_leaf=1, random_state=42, n_jobs=-1)
rf.fit(Xtr_s, ytr)
rfp = rf.predict(Xte_s)
rfpr = rf.predict_proba(Xte_s)[:,1]
rfa = accuracy_score(yte, rfp)
rfauc = roc_auc_score(yte, rfpr)
print(f"   RF  Accuracy: {rfa*100:.2f}%  AUC: {rfauc:.4f}")

joblib.dump(lr, 'models/logistic_regression_model.pkl')
joblib.dump(rf, 'models/random_forest_model.pkl')

print("\n[6/6] Evaluation plots...")
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Model Performance', fontsize=16, fontweight='bold')

cm1 = confusion_matrix(yte, lrp)
sns.heatmap(cm1, annot=True, fmt='d', cmap='Blues', ax=axes[0,0], xticklabels=['Stay','Churn'], yticklabels=['Stay','Churn'])
axes[0,0].set_title(f'LR Confusion Matrix\nAcc: {lra:.2%}')
axes[0,0].set_ylabel('Actual')
axes[0,0].set_xlabel('Predicted')

cm2 = confusion_matrix(yte, rfp)
sns.heatmap(cm2, annot=True, fmt='d', cmap='Greens', ax=axes[0,1], xticklabels=['Stay','Churn'], yticklabels=['Stay','Churn'])
axes[0,1].set_title(f'RF Confusion Matrix\nAcc: {rfa:.2%}')
axes[0,1].set_ylabel('Actual')
axes[0,1].set_xlabel('Predicted')

f1,t1,_ = roc_curve(yte, lrpr)
f2,t2,_ = roc_curve(yte, rfpr)
axes[0,2].plot(f1,t1,'b',lw=2,label=f'LR (AUC={lrauc:.3f})')
axes[0,2].plot(f2,t2,'g',lw=2,label=f'RF (AUC={rfauc:.3f})')
axes[0,2].plot([0,1],[0,1],'gray',linestyle='--')
axes[0,2].set_title('ROC Curves')
axes[0,2].legend()
axes[0,2].grid(True, alpha=0.3)

fi = pd.DataFrame({'f': fnames, 'i': rf.feature_importances_}).sort_values('i').tail(12)
axes[1,0].barh(fi['f'], fi['i'], color='#3498db')
axes[1,0].set_title('Feature Importance')

b = axes[1,1].bar(['LR','RF'], [lra*100, rfa*100], color=['#3498db','#2ecc71'], width=0.4)
axes[1,1].set_title('Accuracy Comparison')
axes[1,1].set_ylim(0, 110)
for bar, v in zip(b, [lra*100, rfa*100]):
    axes[1,1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+1, f'{v:.1f}%', ha='center', fontweight='bold', fontsize=14)

rpt = classification_report(yte, rfp, target_names=['Stay','Churn'], output_dict=True)
rd = [[l, f"{rpt[l]['precision']:.3f}", f"{rpt[l]['recall']:.3f}", f"{rpt[l]['f1-score']:.3f}"] for l in ['Stay','Churn']]
axes[1,2].axis('off')
t = axes[1,2].table(cellText=rd, colLabels=['Class','Precision','Recall','F1'], cellLoc='center', loc='center', bbox=[0,0.3,1,0.5])
t.auto_set_font_size(False)
t.set_fontsize(12)
t.scale(1,2.5)
axes[1,2].set_title('RF Classification Report', fontweight='bold')

plt.tight_layout()
plt.savefig('static/plots/model_evaluation.png', dpi=150, bbox_inches='tight')
plt.close()

print("\n" + "="*60)
print("        TRAINING COMPLETE!")
print("="*60)
print(f"\n  Logistic Regression: {lra*100:.2f}% accuracy, {lrauc:.4f} AUC")
print(f"  Random Forest:       {rfa*100:.2f}% accuracy, {rfauc:.4f} AUC")
print(f"\n  Run next: python app.py")
print("="*60)
print(classification_report(yte, lrp, target_names=['Stay','Churn']))
print(classification_report(yte, rfp, target_names=['Stay','Churn']))