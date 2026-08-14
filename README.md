# My Data Analytics Portfolio

Hi! I'm Kyle — a data analyst with experience in SQL, Python, and data visualization. This repository showcases my analytics projects, focusing on real-world business questions, clean analysis, and clear insights.

## 🔧 Skills

- SQL (PostgreSQL)
- Python (pandas, numpy, matplotlib, seaborn)
- Data Visualization (Tableau, Power BI)
- Excel / Google Sheets
- Data Cleaning & EDA
- Machine Learning

## 📂 Projects

### ✈️ [Airline_Database] (./projects/Airline_Database)
**Type:** SQL database design & querying

Designed and built a fully normalized relational airline database in PostgreSQL simulating real-world operations — aircraft seating, international multi-leg flight scheduling across time zones, group reservations, and seat assignment. Used generate_series() and recursive CTEs to programmatically generate realistic data at scale (~1,200 passengers, weekly flights across 4 international routes, full seat maps), and wrote Faker-powered synthetic data generation via plpython3u. Included analytics queries for profit estimation, seat occupancy, and time-zone-aware scheduling.

**Tools:** PostgreSQL
**Key skills demonstrated:** Schema design, normalization, complex joins, query optimization]

---

### 🏀 [GAT_Basketball_Next_Level_Action](./projects/GAT_Basketball_Next_Level_Action)
**Type:** Machine learning / predictive modeling

Built a Graph Attention Network (GAT) to predict the next event in an NBA possession (e.g., shot, rebound, foul, substitution) using play-by-play data from the 2000–2001 NBA season. Each possession snapshot is modeled as a fully connected graph over the ten on-court players, with node features combining learned player embeddings, lineup/role indicators, offense-defense status, and engineered temporal context (score margin, game clock, possession progress, recent event history). Trained a 3-layer multi-head GAT with PyTorch Geometric and benchmarked it against a Mistral 7B LLM baseline using zero-shot and few-shot prompting on textual play-by-play descriptions.

Results: The GAT substantially outperformed the LLM baseline across all metrics — 52% Top-1 accuracy and 87% Top-3 accuracy vs. under 12% Top-1 for either LLM prompting strategy — demonstrating that explicit relational structure captures basketball dynamics far better than unstructured text-based prediction. A class-weighted training variant improved recall on rare events (ejections, timeouts, violations) at a small cost to overall accuracy.

Tools: Python, PyTorch Geometric, Graph Attention Networks, Mistral 7B (prompting), pandas
Key skills demonstrated: graph neural network design, multi-agent relational modeling, feature engineering, model evaluation (Top-K accuracy, macro-F1, confusion matrix analysis), LLM baselining, academic research/writing

---

### 🐧 [Penguin_Risk](./projects/Penguin_Risk)
**Type:** Machine learning / predictive modeling

[Replace with 2-3 sentences: What risk were you modeling? What dataset did you use (e.g., Palmer Penguins)? What was your approach and result?]

**Tools:** Python (pandas, scikit-learn, matplotlib/seaborn)
**Key skills demonstrated:** [e.g., classification, EDA, model comparison]

---

> 📌 More projects in progress — check back for updates, or reach out below if you'd like to see work in development.

## 📬 Contact

- Email: [use a contact form or LinkedIn instead of plain-text email to avoid spam bots]
- LinkedIn: [add your LinkedIn URL]
- GitHub: [@applekyle5652](https://github.com/applekyle5652)
