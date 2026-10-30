"""
Sample Benchmark Data, Baseline Metrics, and Test Suites for BERT-PhishFinder.
Derived directly from:
1. IEEE TDSC 2025 Paper: "BERT-PhishFinder: A Robust Model for Accurate Phishing URL Detection With Optimized DistilBERT"
2. Yash Bhatia's Reimplementation Project Explainer (October 2026)
"""

PRESET_BENCHMARK_URLS = [
    # 10 Legitimate Real-World URLs
    {
        "id": "legit-1",
        "url": "https://www.google.com/search?q=cybersecurity+research",
        "ground_truth": "Legitimate",
        "category": "Search & Web Portal",
        "description": "Standard encrypted search engine query with standard subdomains and parameters."
    },
    {
        "id": "legit-2",
        "url": "https://github.com/Aljofey/BERT-PhishFinder",
        "ground_truth": "Legitimate",
        "category": "Developer Repository",
        "description": "Official author's code repository referenced in IEEE TDSC 2025 paper."
    },
    {
        "id": "legit-3",
        "url": "https://www.bits-pilani.ac.in/academics/computer-science",
        "ground_truth": "Legitimate",
        "category": "Academic Institution",
        "description": "Official university domain with verified institutional top-level domain (.ac.in)."
    },
    {
        "id": "legit-4",
        "url": "https://en.wikipedia.org/wiki/Phishing",
        "ground_truth": "Legitimate",
        "category": "Educational / Reference",
        "description": "Clean encyclopedic article URL on phishing methodologies."
    },
    {
        "id": "legit-5",
        "url": "https://www.amazon.com/gp/help/customer/display.html?nodeId=508510",
        "ground_truth": "Legitimate",
        "category": "E-Commerce",
        "description": "Legitimate customer support portal for Amazon with standard hierarchical path."
    },
    {
        "id": "legit-6",
        "url": "https://www.paypal.com/signin",
        "ground_truth": "Legitimate",
        "category": "Fintech / Banking",
        "description": "Official authentic PayPal authentication portal (frequently targeted by impersonators)."
    },
    {
        "id": "legit-7",
        "url": "https://login.microsoftonline.com/common/oauth2/authorize",
        "ground_truth": "Legitimate",
        "category": "Enterprise Authentication",
        "description": "Legitimate Microsoft OAuth single-sign-on redirect endpoint."
    },
    {
        "id": "legit-8",
        "url": "https://www.netflix.com/browse",
        "ground_truth": "Legitimate",
        "category": "Entertainment",
        "description": "Official authenticated subscriber landing page."
    },
    {
        "id": "legit-9",
        "url": "https://huggingface.co/distilbert/distilbert-base-uncased",
        "ground_truth": "Legitimate",
        "category": "Machine Learning Hub",
        "description": "Official pretrained model card used as the BERT-PhishFinder backbone."
    },
    {
        "id": "legit-10",
        "url": "https://stackoverflow.com/questions/tagged/deep-learning",
        "ground_truth": "Legitimate",
        "category": "Developer Community",
        "description": "Standard developer forum category listing."
    },

    # 10 Phishing Real-World URLs (Deceptive structures, typosquatting, subdomains, odd TLDs)
    {
        "id": "phish-1",
        "url": "http://paypal-security-update-verification.com/login.php?client=auth",
        "ground_truth": "Phishing",
        "category": "Brand Impersonation (PayPal)",
        "description": "Hyphenated keyword stuffing mimicking PayPal account verification to harvest credentials."
    },
    {
        "id": "phish-2",
        "url": "http://apple-id-verify-icloud.account-alert.xyz/auth/login",
        "ground_truth": "Phishing",
        "category": "Credential Harvester (Apple)",
        "description": "Spoofed Apple ID recovery portal hosted on cheap/suspicious .xyz TLD with nested subdomains."
    },
    {
        "id": "phish-3",
        "url": "http://192.168.1.105/chase-online-banking-secure/signin.html",
        "ground_truth": "Phishing",
        "category": "Raw IP Phishing",
        "description": "Direct numerical IP address hosting a fake Chase online banking login page."
    },
    {
        "id": "phish-4",
        "url": "https://netflix-subscription-renew-billing.top/update-payment",
        "ground_truth": "Phishing",
        "category": "Financial Scam (Netflix)",
        "description": "Fake payment update phishing page exploiting fear of account suspension."
    },
    {
        "id": "phish-5",
        "url": "http://secure-wellsfargo-bank.com.login-verify.info/online-access",
        "ground_truth": "Phishing",
        "category": "Subdomain Trickery",
        "description": "Legitimate domain name embedded inside subdomain ('wellsfargo-bank.com') with actual host at .info."
    },
    {
        "id": "phish-6",
        "url": "https://google-docs-drive-share-verify.tk/document/d/edit",
        "ground_truth": "Phishing",
        "category": "Cloud Document Phishing",
        "description": "Malicious clone of Google Docs viewer on a free/abuse-prone .tk TLD."
    },
    {
        "id": "phish-7",
        "url": "http://binance-wallet-airdrop-claim.live/connect-wallet",
        "ground_truth": "Phishing",
        "category": "Crypto Drainer Scam",
        "description": "Fake cryptocurrency wallet connect trap targeting Web3 private keys."
    },
    {
        "id": "phish-8",
        "url": "http://microsoft-office365-password-reset.buzz/portal",
        "ground_truth": "Phishing",
        "category": "Enterprise Spoofing",
        "description": "Corporate email harvesting page disguised as an urgent Office 365 password reset alert."
    },
    {
        "id": "phish-9",
        "url": "http://login-amazon-prime-membership-renew.club/verify-card",
        "ground_truth": "Phishing",
        "category": "E-Commerce Phishing",
        "description": "Typosquatted e-commerce refund scam on .club TLD with credential extraction form."
    },
    {
        "id": "phish-10",
        "url": "http://instagram-help-copyright-infringement-appeal.gq/confirm",
        "ground_truth": "Phishing",
        "category": "Social Media Spear-Phish",
        "description": "Bogus copyright violation notification seeking Instagram account takeover."
    }
]

# 30 Testing URLs added in phishfinder.ipynb
CUSTOM_TEST_URLS = [
    "https://app-mercadopromo.com/safe.php?id=142",
    "https://secure-login.sbi-verification.example/auth/login",
    "https://hdfc-netbanking-update.example/customer/verify",
    "https://icici-account-security.example/login",
    "https://paypal-resolution-center.example/verify",
    "https://paytm-kyc-update.example/complete-verification",
    "https://upi-payment-confirmation.example/transaction/verify",
    "https://accounts-google-security.example/signin",
    "https://login-microsoft365-verify.example/auth",
    "https://instagram-account-recovery.example/login",
    "https://facebook-security-alert.example/checkpoint",
    "https://outlook-mail-verification.example/session",
    "https://github-security-check.example/login/oauth",
    "https://amazon-order-confirmation.example/orders/track",
    "https://flipkart-refund-processing.example/claim",
    "https://dhl-shipment-update.example/tracking",
    "https://fedex-delivery-fee.example/payment",
    "https://india-post-parcel.example/address/update",
    "https://accounts.google.com.security-check.example/login",
    "https://login.microsoftonline.com.verify-session.example/auth",
    "https://secure-paypal-com.example/signin?session=839204",
    "https://customer-support.example/verify?redirect=account",
    "https://auth-service.example/login?continue=%2Fdashboard",
    "https://billing-update.example/invoice?id=INV-20261030",
    "http://192.0.2.45/login/verify",
    "http://198.51.100.23/secure/account",
    "https://xn--paypl-3ve.example/signin",
    "https://secure-login.example@authentication-check.example/login",
    "https://account-validation.example/login.php?token=83920",
    "https://signin.example/verify?email=user%40example.com"
]

# Paper Benchmark Performance Statistics (Tables I, II, III, IV, V, VI from IEEE TDSC 2025)
PAPER_METRICS = {
    "datasets": [
        {"name": "DS1 (Ebbu2017)", "total": 73575, "phishing": 37175, "legitimate": 36400, "source": "GitHub (Ebbu2017)"},
        {"name": "DS2", "total": 126077, "phishing": 40668, "legitimate": 85409, "source": "CatchPhish & Common-crawl"},
        {"name": "DS3", "total": 82888, "phishing": 40668, "legitimate": 42220, "source": "PhishTank & Common-crawl"},
        {"name": "DS4", "total": 83857, "phishing": 40668, "legitimate": 43189, "source": "PhishTank & Alexa"},
        {"name": "DS5 (Kaggle)", "total": 549346, "phishing": 156422, "legitimate": 392924, "source": "Kaggle (Tiwari 2019)"},
        {"name": "My Version (Kaggle Mini)", "total": 30000, "phishing": 15000, "legitimate": 15000, "source": "Kaggle 30k Stratified Split"}
    ],
    "bert_phishfinder_cv": [
        {"dataset": "DS1", "accuracy": 99.26, "precision": 99.27, "recall": 99.25, "f1": 99.26, "auc": 99.26},
        {"dataset": "DS2", "accuracy": 96.68, "precision": 96.31, "recall": 93.32, "f1": 94.78, "auc": 95.85},
        {"dataset": "DS3", "accuracy": 96.80, "precision": 96.73, "recall": 96.74, "f1": 96.73, "auc": 96.77},
        {"dataset": "DS4", "accuracy": 96.95, "precision": 97.49, "recall": 96.18, "f1": 96.83, "auc": 96.92},
        {"dataset": "DS5", "accuracy": 99.16, "precision": 98.84, "recall": 98.22, "f1": 98.53, "auc": 98.88}
    ],
    "ds1_confusion_matrices": {
        "Fold 1": {"samples": 14677, "tp": 7337, "tn": 7230, "fp": 56, "fn": 54, "acc": 99.25},
        "Fold 2": {"samples": 14677, "tp": 7340, "tn": 7228, "fp": 58, "fn": 51, "acc": 99.26},
        "Fold 3": {"samples": 14677, "tp": 7330, "tn": 7240, "fp": 46, "fn": 60, "acc": 99.27},
        "Fold 4": {"samples": 14677, "tp": 7335, "tn": 7232, "fp": 54, "fn": 55, "acc": 99.25},
        "Fold 5": {"samples": 14677, "tp": 7334, "tn": 7233, "fp": 53, "fn": 56, "acc": 99.26}
    },
    "efficiency_benchmarks": [
        {"model": "BERT-PhishFinder", "seconds_per_epoch": 400, "inference_ms": 0.47, "param_type": "DistilBERT + 8 Dense Heads"},
        {"model": "ALBERT", "seconds_per_epoch": 945, "inference_ms": 0.98, "param_type": "albert-base-v2"},
        {"model": "XLNet", "seconds_per_epoch": 1800, "inference_ms": 1.85, "param_type": "xlnet-base-cased"},
        {"model": "RoBERTa", "seconds_per_epoch": 480, "inference_ms": 0.58, "param_type": "distilroberta-base"}
    ]
}

# Baseline Comparisons on DS1 (from Table IV and Table V)
BASELINE_COMPARISONS = [
    {"model": "CNN-LSTM (Shi et al.)", "accuracy": 98.62, "precision": 98.82, "recall": 98.45, "f1": 98.63, "type": "Deep Learning Baseline"},
    {"model": "TCN (Bai et al.)", "accuracy": 98.80, "precision": 98.73, "recall": 98.90, "f1": 98.81, "type": "Temporal ConvNet"},
    {"model": "ALBERT (Lan et al.)", "accuracy": 98.74, "precision": 98.60, "recall": 98.92, "f1": 98.76, "type": "Transformer"},
    {"model": "Multi-Scale CNN (Hussain et al.)", "accuracy": 98.89, "precision": 98.85, "recall": 98.96, "f1": 98.90, "type": "Deep Learning Baseline"},
    {"model": "XLNet (Yang et al.)", "accuracy": 99.14, "precision": 99.07, "recall": 99.23, "f1": 99.15, "type": "Autoregressive Transformer"},
    {"model": "RoBERTa (Liu et al.)", "accuracy": 99.10, "precision": 99.02, "recall": 99.21, "f1": 99.11, "type": "Transformer"},
    {"model": "BERT-PhishFinder (Proposed)", "accuracy": 99.26, "precision": 99.27, "recall": 99.25, "f1": 99.26, "type": "Ours (DistilBERT + Dual Pool + 8 Heads)"}
]
