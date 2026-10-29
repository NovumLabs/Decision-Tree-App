\# Novum Labs Decision Tree AI



\*\*Version:\*\* 0.2.0

\*\*Organization:\*\* Novum Labs Pvt. Limited

\*\*Repository:\*\* https://github.com/NovumLabs/Decision-Tree-App



An educational, mathematically grounded Python project for developing an explainable Decision Tree machine-learning engine from first principles.



\## Division I — Mathematical Criteria Foundation



Division I establishes the mathematical criteria and testing foundation required for further Decision Tree development.



\### Implemented and tested



\- \*\*Entropy:\*\* measures uncertainty in a class distribution.

\- \*\*Gini Impurity:\*\* measures class impurity.

\- \*\*Information Gain:\*\* measures the reduction in entropy produced by a split.

\- \*\*Numeric split support:\*\* an initial step toward selecting useful splits.

\- \*\*Input validation:\*\* tests cover invalid probability distributions, empty inputs, and invalid group sizes.

\- \*\*Automated testing:\*\* the current unit test suite contains 28 passing tests.



These results describe the current tested implementation. They do not mean that the complete Decision Tree engine or application is finished.



\## Mathematical Foundation



For class probabilities \\(p\_1, p\_2, \\ldots, p\_k\\), entropy is defined as:



\\\[

H(S)=-\\sum\_{i=1}^{k}p\_i\\log\_2(p\_i)

\\]



Gini Impurity is defined as:



\\\[

G(S)=1-\\sum\_{i=1}^{k}p\_i^2

\\]



For a parent dataset \\(S\\) divided into child groups \\(S\_j\\), Information Gain is:



\\\[

IG(S)=H(S)-\\sum\_j\\frac{|S\_j|}{|S|}H(S\_j)

\\]



These criteria help evaluate the quality of candidate splits during Decision Tree construction.



\## Development Roadmap



\### Division I — Mathematical Criteria Foundation



\- Implement and test impurity criteria.

\- Validate probability distributions and edge cases.

\- Develop and test split-selection functionality.

\- Document mathematical definitions, examples, and limitations.



\### Division II — Decision Tree Engine



\- Develop classification and regression trees from scratch.

\- Add stopping conditions and prediction.

\- Add tree inspection and reproducible examples.

\- Compare results against appropriate scikit-learn baselines.



\### Division III — Advanced Features



\- Investigate pre-pruning and post-pruning.

\- Handle missing values, categorical features, class weights, and imbalanced datasets.

\- Explore Random Forest as an extension.

\- Develop decision paths, human-readable rules, feature importance, and tree visualization.



\### Division IV — Application and Delivery



\- Evaluate a FastAPI backend and React with TypeScript frontend.

\- Improve packaging, documentation, examples, and deployment.

\- Introduce continuous integration and a release checklist.



Divisions II–IV represent planned development, not claims of completed functionality.



\## Testing



Run the unit test suite from the project root:



```powershell

.\\.venv\\Scripts\\python.exe -m pytest -v

```



At the time of preparing this release, the suite has \*\*28 passing tests\*\*. Run the command again to verify the result in your environment.



\## Technology



\- Python 3.10 or later

\- NumPy

\- pandas

\- pytest

\- scikit-learn, Jupyter, matplotlib, and Ruff as development extras



\## Project Status



\*\*Release:\*\* v0.2.0 — Division I: Mathematical Criteria Foundation



This is an actively developed educational machine-learning project. The current release focuses on mathematical criteria, initial split-selection functionality, and automated tests. A complete production-ready Decision Tree application remains a future goal.



\## Maintained by



\*\*Novum Labs Pvt. Limited\*\*



Technology for a Better World.
