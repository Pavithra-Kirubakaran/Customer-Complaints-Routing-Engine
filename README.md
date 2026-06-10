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

## FastAPI Agentic Routing API

### Run the API

```bash
uvicorn main:app --reload
```

Open:

```text
http://localhost:8000/docs
```

### Sample request

```bash
curl -X POST "http://localhost:8000/tickets" \
  -H "Content-Type: application/json" \
  -d '{
    "customer_id": "cust-001",
    "channel": "email",
    "subject": "Duplicate charge on my card",
    "message": "I was charged twice for my last order and need a refund."
  }'
```

### New files and flow

- `main.py` - FastAPI server entry point
- `agents/` - supervisor, category, priority, RAG, SLA, routing, monitoring, escalation agents
- `database.py` - SQLite storage for ticket history and monitoring
- `data/knowledge_base.json` - local knowledge store for the RAG agent

This new flow preserves the existing `app.py` Streamlit interface while adding a production-ready API layer and agent orchestration.
