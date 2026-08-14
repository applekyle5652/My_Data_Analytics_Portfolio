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

### ✈️ [Airline_Database](./projects/Airline_Database)
**Type:** SQL database design & querying

Designed and built a fully normalized relational airline database in PostgreSQL simulating real-world operations — aircraft seating, international multi-leg flight scheduling across time zones, group reservations, and seat assignment. Used generate_series() and recursive CTEs to programmatically generate realistic data at scale (~1,200 passengers, weekly flights across 4 international routes, full seat maps), and wrote Faker-powered synthetic data generation via plpython3u. Included analytics queries for profit estimation, seat occupancy, and time-zone-aware scheduling.

**Tools:** PostgreSQL

**Key skills demonstrated:** Schema design, normalization, complex joins, query optimization]

---

### 🏀 [GAT_Basketball_Next_Level_Action](./projects/GAT_Basketball_Next_Level_Action)
**Type:** Machine learning / predictive modeling

Built a Graph Attention Network (GAT) to predict the next event in an NBA possession (e.g., shot, rebound, foul, substitution) using play-by-play data from the 2000–2001 NBA season. Each possession snapshot is modeled as a fully connected graph over the ten on-court players, with node features combining learned player embeddings, lineup/role indicators, offense-defense status, and engineered temporal context (score margin, game clock, possession progress, recent event history). Trained a 3-layer multi-head GAT with PyTorch Geometric and benchmarked it against a Mistral 7B LLM baseline using zero-shot and few-shot prompting on textual play-by-play descriptions.

Results: The GAT substantially outperformed the LLM baseline across all metrics — 52% Top-1 accuracy and 87% Top-3 accuracy vs. under 12% Top-1 for either LLM prompting strategy — demonstrating that explicit relational structure captures basketball dynamics far better than unstructured text-based prediction. A class-weighted training variant improved recall on rare events (ejections, timeouts, violations) at a small cost to overall accuracy.

**Tools:** Python, PyTorch Geometric, Graph Attention Networks, Mistral 7B (prompting), pandas

**Key skills demonstrated:** graph neural network design, multi-agent relational modeling, feature engineering, model evaluation (Top-K accuracy, macro-F1, confusion matrix analysis), LLM baselining, academic research/writing

---

### 🐧 [Penguin_Risk](https://storymaps.arcgis.com/stories/a74ed40b3658492494ab44020297eacd)
**Type:** GIS spatial analysis — ArcGIS StoryMap

Built a spatial risk index to identify which emperor penguin breeding colonies across Antarctica are most vulnerable to environmental and human stressors, combining colony location data (MAPPPD), sea-ice concentration rasters (U.S. National Ice Center), and human activity layers — research stations, camps, routes (Australian Antarctic Data Centre). Used raster calculation and reclassification in ArcGIS to build two component indices — an ice-stability risk surface (scored 0–1 by ice type: fast, multi-year, first-year, thin, open water) and a distance-based human disturbance risk surface — then combined them into a weighted composite risk index per colony using zonal statistics, with sea-ice risk weighted more heavily as the dominant ecological driver.

**Results:** Sea-ice instability emerged as the primary risk driver continent-wide, with a subset of colonies showing compounded risk where unstable ice overlaps with proximity to research stations or travel routes — flagging specific sites for conservation priority.

**Tools:** ArcGIS (raster calculator, zonal statistics, distance accumulation), Antarctic Polar Stereographic projection (EPSG:3031), ArcGIS StoryMaps (interactive dashboard)

**Key skills demonstrated:** spatial risk modeling, raster analysis, multi-source geospatial data integration, GIS projection handling, interactive dashboard design, technical writing
---

> 📌 More projects in progress — check back for updates, or reach out below if you'd like to see work in development.

## 📬 Contact

- Email: kyleng5652@gmail.com
- LinkedIn: www.linkedin.com/in/kyleng5652
- GitHub: [@applekyle5652](https://github.com/applekyle5652)
