# Reaction equilibration framework

the overall project structure is as follows:

```
reaction_equilibration/
├── template_extraction/ #used to extract templates from chemical reactions
└── uspto_balance/ #creates in-silico validated reactions for given retrosynthetic template and pool of molecules
```
## Project installation 

```
cd to/the/target/folder
git clone -b refactor git@github.com:yvsgrndjn/USPTO_balance.git
```
## Two possibities:

### 1. if you want to start by extracting the retrosynthetic templates from your set of reactions:
```
cd reaction_equilibration/template_extraction/
```
and follow instructions there for environment installation

### 2. if you already have the templates you are wishing to enrich, skip directly to the uspto_balance folder
```
cd reaction_equilibration/uspto_balance/
```
 and follow instructions there for environment installation
