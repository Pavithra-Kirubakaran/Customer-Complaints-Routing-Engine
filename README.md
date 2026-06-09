## How to Run the Project

### 1. Clone the Repository

```bash
git clone https://github.com/sanashaikh38/Customer-Complaints-Routing-Engine
cd Customer-Complaints-Routing-Engine
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

### 3. Activate the Virtual Environment

#### Windows

```bash
venv\Scripts\activate
```

#### Linux / Mac

```bash
source venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the Streamlit Application

```bash
streamlit run app.py
```

### 6. Open the Application

After running the above command, open:

```text
http://localhost:8501
```

in your browser.

---

## Project Structure

```text
customer-complaint-routing-engine/
│
├── models/
│   ├── queue_model.pkl
│   └── priority_model.pkl
│
├── notebook/
│   └── complaint_classification.ipynb
│
├── app.py
├── routing.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Sample Complaint

```text
My payment was deducted twice but my order was not placed.
Please refund the amount immediately.
```

### Sample Output

```text
Queue: Billing and Payments
Priority: High
Assigned Team: Finance Team
SLA: 4 Hours
```
