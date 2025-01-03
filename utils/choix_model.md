# Choix du Meilleur Modèle :

Pour déterminer un meilleur modèle, j'ai entrainé plusieurs modèles de la version **V0** à la version **V6**. Je vais faire une comparaison entre les deux meilleurs **V5** et **V6**. Nous allons examiner leurs performances en nous basant sur les graphiques, matrices de confusion et rapports de classification.

---

## Analyse des Graphiques de Précision et Pertes

**V5 :**
- Précision d'apprentissage : **98.5%**
- Perte d'apprentissage' : **4.6%**
- Précision de test : **99.3%**
- Perte de test : **2.5%**

**V6 :**
- Précision d'apprentissage : **98.8%**
- Perte d'apprentissage' : **3.7%**
- Précision de test : **99.0%**
- Perte de test : **3.4%**


**V5** a une meilleure précision sur le test (**99.3% vs. 99.0%**) et une perte plus faible (**2.5% vs. 3.4%**). Cela suggère que **V5** généralise légèrement mieux sur le test. Cependant, pour l'apprentissage **V6** est similairement mieux.

---

## Analyse des Matrices de Confusion

**V5 :**
- La matrice montre un alignement diagonal clair, avec peu d'erreurs hors de la diagonale.
- Les classes comme "9", "A", "S" et "prière" ont des performances correctes, mais quelques confusions mineures apparaissent.

**V6 :**
- La matrice est également bien alignée sur la diagonale, mais les confusions dans certaines classes, comme "E", "S" et "T", sont légèrement plus fréquentes comparé à **V5**.


La matrice de confusion de **V5** montre des performances légèrement plus cohérentes, en particulier pour les classes problématiques.

---

## Analyse des Rapports de Classification

### Précision :
- Les deux modèles ont une précision moyenne pondérée de **99%**.
- Cependant, **V5** a une meilleure précision pour des classes spécifiques, comme "9" (**0.86 vs. 0.90**) et "E" (**1.00 vs. 0.80**).

### Rappel :
- Le rappel est légèrement meilleur pour **V5** dans certaines classes problématiques, comme "S" et "T".

### F1-Score :
- Le F1-score est globalement similaire, mais **V5** montre une meilleure stabilité dans les classes ayant moins de données.

---

Sur la base des analyses :
- **Précision du test :** V5 est meilleur (**99.3% vs. 99.0%**).
- **Perte :** V5 a une perte plus faible.
- **Matrices de confusion :** V5 montre des performances plus cohérentes sur des classes problématiques.
- **Rapports de classification :** V5 a un avantage pour des classes spécifiques.

## Conclusion :
**Le modèle V5 est le meilleur choix** en termes de performances globales et de généralisation sur les données de test.

La différence entre les deux modèles **V5** et **V6** est que **epochs=15** pour **V5** alors que pour **V6** **epochs=20**.