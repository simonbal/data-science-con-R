---
title: "prestamos"
author: "Simon Ballesteros"
date: "17 de junio de 2018"
output: 
  html_document: 
    keep_md: yes
---



## Árboles de decisión (C5.0)
### Identificación de riesgos en préstamos bancarios

Desde la crísis financiera mundial de 2007-2008 se puso de relieve la importancia de la transparencia y el rigor en las prácticas bancarias. Dada la escasez del crédito, los bancos han ajustado sus sistemas de préstamos y han mirado hacia el *machine learning* para identificar de forma mas precisa los riesgos en préstamos.

Los árboles de decisión son ampliamente usados en la industria financiera debido a su alta precisión y a su capacidad de formular un modelo estadístico en lenguaje común. Debido a que los gobiernos supervisan cuidadosamente las précticas crediticias, los ejecutivos deben ser capaces de explicar porqué fue rechazado un crédito a un cliente mientras que le fue aprobado a otro. Esta información es útil también para el cliente quien puede determinar si su calificación de riesgo es o no satisfactoria.

En este artículo desarrollamos un modelo sencillo de aprobación de créditos usando árboles de decisión C5.0. También veremos como ajustar los resultados del modelo para minimizar errores que puedan resultar en pérdidas financieras para la institución.

### Paso 1 - Recolección de datos

La idea del modelo de crédito es identificar los factores que son predictivos de un alto riesgo de incumplimiento. Por lo tanto, se necesita recolectar un gran número de préstamos bancarios anteriores, e identificar los que fueron objeto de incumplimiento, así como información sobre los solicitantes.

Un conjunto de datos con estas características fue donado por el UCI Machine Learning Data Repository (http://archive.ics.uci.edu/ml) por Hans Hofmann de la universidad de Hamburgo. El conjunto de datos contiene información sobre préstamos obtenidos de una entidad crediticia alemana.

El archivo contiene 1.000 ejemplos de créditos, más un conjunto de características numéricas acerca del préstamo y del cliente. Una variable *class* indica si el préstamo fue impagado. Veamos si podemos determinar patrones que produzcan este resultado.

### Paso 2 - Exploración y preparación de los datos

Importaremos el archivo con *read.csv()* e ignoraremos la opción *stringsAsFactor*, usando su opcion por defecto TRUE, ya que la mayoría de las características de los datos son nominales:


```r
credito <- read.csv("credit.csv")
```
Las primeras líneas que genera la función *str()* son:

```r
str(credito)
```

```
## 'data.frame':	1000 obs. of  17 variables:
##  $ checking_balance    : Factor w/ 4 levels "< 0 DM","> 200 DM",..: 1 3 4 1 1 4 4 3 4 3 ...
##  $ months_loan_duration: int  6 48 12 42 24 36 24 36 12 30 ...
##  $ credit_history      : Factor w/ 5 levels "critical","good",..: 1 2 1 2 4 2 2 2 2 1 ...
##  $ purpose             : Factor w/ 6 levels "business","car",..: 5 5 4 5 2 4 5 2 5 2 ...
##  $ amount              : int  1169 5951 2096 7882 4870 9055 2835 6948 3059 5234 ...
##  $ savings_balance     : Factor w/ 5 levels "< 100 DM","> 1000 DM",..: 5 1 1 1 1 5 4 1 2 1 ...
##  $ employment_duration : Factor w/ 5 levels "< 1 year","> 7 years",..: 2 3 4 4 3 3 2 3 4 5 ...
##  $ percent_of_income   : int  4 2 2 2 3 2 3 2 2 4 ...
##  $ years_at_residence  : int  4 2 3 4 4 4 4 2 4 2 ...
##  $ age                 : int  67 22 49 45 53 35 53 35 61 28 ...
##  $ other_credit        : Factor w/ 3 levels "bank","none",..: 2 2 2 2 2 2 2 2 2 2 ...
##  $ housing             : Factor w/ 3 levels "other","own",..: 2 2 2 1 1 1 2 3 2 2 ...
##  $ existing_loans_count: int  2 1 1 1 2 1 1 1 1 2 ...
##  $ job                 : Factor w/ 4 levels "management","skilled",..: 2 2 4 2 2 4 2 1 4 1 ...
##  $ dependents          : int  1 1 2 2 2 2 1 1 1 1 ...
##  $ phone               : Factor w/ 2 levels "no","yes": 2 1 1 1 1 2 1 2 1 1 ...
##  $ default             : Factor w/ 2 levels "no","yes": 1 2 1 1 2 1 1 1 1 2 ...
```
Vemos las 1.000 observaciones esperadas y sus 17 características, que incluyen tipos de datos *factor* y *entero*.

Veamos la salida de *table()* para un par de características que podrían predecir el impago. El saldo de cuenta (checking_balance) y el saldo de ahorros (savings_balance) del solicitante se guardaron como variables categóricas:

```r
table(credito$checking_balance)
```

```
## 
##     < 0 DM   > 200 DM 1 - 200 DM    unknown 
##        274         63        269        394
```

```r
table(credito$savings_balance)
```

```
## 
##      < 100 DM     > 1000 DM  100 - 500 DM 500 - 1000 DM       unknown 
##           603            48           103            63           183
```
Los balances de cuenta podrían probar ser importantes predictores del estado de impago de los créditos. Nótese que dado que los datos fueron obtenidos de Alemania, la moneda es el marco alemán (Deutsche Mark - DM).

Algunas de las características del préstamo son numéricas, como la duración del préstamo y la cantidad solicitada.

```r
summary(credito$months_loan_duration)
```

```
##    Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
##     4.0    12.0    18.0    20.9    24.0    72.0
```

```r
summary(credito$amount)
```

```
##    Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
##     250    1366    2320    3271    3972   18424
```
Las cantidades prestadas van de 250 DM a 18.424 DM y entre 4 y 72 meses, con una duración media de 18 meses y promedio de 2.320 DM.

El vector *default* indica si el aplicante del crédito fue incapaz de cumplir con los términos del préstamo y cayó en impago. El 30 porciento de los préstamos del conjunto fueron imapagados.


```r
table(credito$default)
```

```
## 
##  no yes 
## 700 300
```
Una alta taza de impagos es indeseable para un banco, ya que implica que la entidad no recupere en su totalidad la inversión. Si tenemos éxito nuestro modelo identificará solicitantes con un alto riesto de impago, permitiento al banco rechazar las solicitudes.

#### Preparación de los datos
##### Creación de conjuntos aleatorios de datos para entrenamiento y prueba

Partiremos el conjunto de datos en dos: un conjunto de entrenamiento para construir el arbol de decisión y un conjunto de prueba para evaluar el rendimiento del modelo sobre datos nuevos. Usaremos 90% de datos para entrenamientto y 10% para prueba.

Los datos no estan aleatoriamente distribuidos en el conjunto. Suponiendo que el banco ordenó los datos ascendentemente por cantidad del préstamo, nos hallaríamos entrenando el modelo con pequeños créditos y probándolo con los mayores. Obviamente esto es problematico.

Resolvemos el problema usando una muestra aleatoria de los datos para entrenamiento. Este es un proceso que selecciona al azar un subconjunto de los datos. En **R**, esto se consigue usando la función *sample()*. Es una práctica común iniciar con un valor semilla (seed), que obliga al proceso de aleatorización a seguir una secuencia que pueda ser replicada posteriormente. 

Las siguientes instrucciones usan la función *sample()* para seleccionar 900 valores al azar dentro de una secuencia de 1 a 1000. Usaremos el valor arbitrario *123* como semilla. Si se omite la semilla, el proceso de aleatorización diferirá cada vez que se ejecute la función. 


```r
set.seed(123)
muestra_entrenamiento <- sample(1000, 900)
```
Según lo esperado, el objeto *muestra_entrenamiento* es un vector de 900 enteros aleatorios.

```r
str(muestra_entrenamiento)
```

```
##  int [1:900] 288 788 409 881 937 46 525 887 548 453 ...
```
Usaremos este vector para seleccionar filas del archivo de crédito, y dividirlo en dos conjuntos: 90% para entrenamiento y 10% para prueba.

```r
credito_train <- credito[muestra_entrenamiento, ]
credito_test <- credito[-muestra_entrenamiento, ]
```
Si todo marchó bien, tendremos cerca del 30% de créditos impagados en cada conjunto de datos.

```r
prop.table(table(credito_train$default))
```

```
## 
##        no       yes 
## 0.7033333 0.2966667
```

```r
prop.table(table(credito_test$default))
```

```
## 
##   no  yes 
## 0.67 0.33
```
Hemos encontrado una particion bastante justa. Ahora podemos construir nuestro arbol de decisión.

### Paso 3 - Entrenamiento del modelo con los datos

Usaremos el algoritmo C5.0 del paquete *C50* para entrenar nuestro modelo de árbol de decisión. Si no lo ha hecho, instale el paquete con *install.packages("C50")* y cárguelo en su sesión de R con *library(C50)*.

Para la primera iteración del modelo, usaremos la configuración por defecto de *C5.0*. La columna 17 del conjunto credito_train es la variable de clase *default*, y necesitamos excluírla del data frame de entrenamiento, pero agregándola como el vector de factores objetivo (target) para la clasificación.

```r
install.packages("C50")
```

```
## Installing package into '/home/simon/R/x86_64-pc-linux-gnu-library/3.4'
## (as 'lib' is unspecified)
```

```r
library(C50)
credito_modelo <- C5.0(credito_train[-17], credito_train$default)
```
El objeto *credito_modelo* contiene ahora un árbol de decisión C5.0. Podemos ver algunos datos básicos del árbol escribiendo su nombre:

```r
credito_modelo
```

```
## 
## Call:
## C5.0.default(x = credito_train[-17], y = credito_train$default)
## 
## Classification Tree
## Number of samples: 900 
## Number of predictors: 16 
## 
## Tree size: 57 
## 
## Non-standard options: attempt to group attributes
```
El texto anterior muestra hechos simples sobre el árbol, incluyendo la función *call* que lo generó, el número de características (*predictors*), y ejemplos (*samples*) usados para poblar el árbol. También se muestra el tamaño del árbol, que indica la profundidad de las decisiones (57).

Para ver las decisiones del árbol, se llama a la función *summary()* en el modelo:

```r
summary(credito_modelo)
```

```
## 
## Call:
## C5.0.default(x = credito_train[-17], y = credito_train$default)
## 
## 
## C5.0 [Release 2.07 GPL Edition]  	Tue Jun 19 18:49:41 2018
## -------------------------------
## 
## Class specified by attribute `outcome'
## 
## Read 900 cases (17 attributes) from undefined.data
## 
## Decision tree:
## 
## checking_balance in {> 200 DM,unknown}: no (412/50)
## checking_balance in {< 0 DM,1 - 200 DM}:
## :...credit_history in {perfect,very good}: yes (59/18)
##     credit_history in {critical,good,poor}:
##     :...months_loan_duration <= 22:
##         :...credit_history = critical: no (72/14)
##         :   credit_history = poor:
##         :   :...dependents > 1: no (5)
##         :   :   dependents <= 1:
##         :   :   :...years_at_residence <= 3: yes (4/1)
##         :   :       years_at_residence > 3: no (5/1)
##         :   credit_history = good:
##         :   :...savings_balance in {> 1000 DM,500 - 1000 DM}: no (15/1)
##         :       savings_balance = 100 - 500 DM:
##         :       :...other_credit = bank: yes (3)
##         :       :   other_credit in {none,store}: no (9/2)
##         :       savings_balance = unknown:
##         :       :...other_credit = bank: yes (1)
##         :       :   other_credit in {none,store}: no (21/8)
##         :       savings_balance = < 100 DM:
##         :       :...purpose in {business,car0,renovations}: no (8/2)
##         :           purpose = education:
##         :           :...checking_balance = < 0 DM: yes (4)
##         :           :   checking_balance = 1 - 200 DM: no (1)
##         :           purpose = car:
##         :           :...employment_duration = > 7 years: yes (5)
##         :           :   employment_duration = unemployed: no (4/1)
##         :           :   employment_duration = < 1 year:
##         :           :   :...years_at_residence <= 2: yes (5)
##         :           :   :   years_at_residence > 2: no (3/1)
##         :           :   employment_duration = 1 - 4 years:
##         :           :   :...years_at_residence <= 2: yes (2)
##         :           :   :   years_at_residence > 2: no (6/1)
##         :           :   employment_duration = 4 - 7 years:
##         :           :   :...amount <= 1680: yes (2)
##         :           :       amount > 1680: no (3)
##         :           purpose = furniture/appliances:
##         :           :...job in {management,unskilled}: no (23/3)
##         :               job = unemployed: yes (1)
##         :               job = skilled:
##         :               :...months_loan_duration > 13: [S1]
##         :                   months_loan_duration <= 13:
##         :                   :...housing in {other,own}: no (23/4)
##         :                       housing = rent:
##         :                       :...percent_of_income <= 3: yes (3)
##         :                           percent_of_income > 3: no (2)
##         months_loan_duration > 22:
##         :...savings_balance = > 1000 DM: no (2)
##             savings_balance = 500 - 1000 DM: yes (4/1)
##             savings_balance = 100 - 500 DM:
##             :...credit_history in {critical,poor}: no (14/3)
##             :   credit_history = good:
##             :   :...other_credit = bank: no (1)
##             :       other_credit in {none,store}: yes (12/2)
##             savings_balance = unknown:
##             :...checking_balance = 1 - 200 DM: no (17)
##             :   checking_balance = < 0 DM:
##             :   :...credit_history = critical: no (1)
##             :       credit_history in {good,poor}: yes (12/3)
##             savings_balance = < 100 DM:
##             :...months_loan_duration > 47: yes (21/2)
##                 months_loan_duration <= 47:
##                 :...housing = other:
##                     :...percent_of_income <= 2: no (6)
##                     :   percent_of_income > 2: yes (9/3)
##                     housing = rent:
##                     :...other_credit = bank: no (1)
##                     :   other_credit in {none,store}: yes (16/3)
##                     housing = own:
##                     :...employment_duration = > 7 years: no (13/4)
##                         employment_duration = 4 - 7 years:
##                         :...job in {management,skilled,
##                         :   :       unemployed}: yes (9/1)
##                         :   job = unskilled: no (1)
##                         employment_duration = unemployed:
##                         :...years_at_residence <= 2: yes (4)
##                         :   years_at_residence > 2: no (3)
##                         employment_duration = 1 - 4 years:
##                         :...purpose in {business,car0,education}: yes (7/1)
##                         :   purpose in {furniture/appliances,
##                         :   :           renovations}: no (7)
##                         :   purpose = car:
##                         :   :...years_at_residence <= 3: yes (3)
##                         :       years_at_residence > 3: no (3)
##                         employment_duration = < 1 year:
##                         :...years_at_residence > 3: yes (5)
##                             years_at_residence <= 3:
##                             :...other_credit = bank: no (0)
##                                 other_credit = store: yes (1)
##                                 other_credit = none:
##                                 :...checking_balance = 1 - 200 DM: no (8/2)
##                                     checking_balance = < 0 DM:
##                                     :...job in {management,skilled,
##                                         :       unemployed}: yes (2)
##                                         job = unskilled: no (3/1)
## 
## SubTree [S1]
## 
## employment_duration in {< 1 year,4 - 7 years}: no (4)
## employment_duration in {> 7 years,1 - 4 years,unemployed}: yes (10)
## 
## 
## Evaluation on training data (900 cases):
## 
## 	    Decision Tree   
## 	  ----------------  
## 	  Size      Errors  
## 
## 	    56  133(14.8%)   <<
## 
## 
## 	   (a)   (b)    <-classified as
## 	  ----  ----
## 	   598    35    (a): class no
## 	    98   169    (b): class yes
## 
## 
## 	Attribute usage:
## 
## 	100.00%	checking_balance
## 	 54.22%	credit_history
## 	 47.67%	months_loan_duration
## 	 38.11%	savings_balance
## 	 14.33%	purpose
## 	 14.33%	housing
## 	 12.56%	employment_duration
## 	  9.00%	job
## 	  8.67%	other_credit
## 	  6.33%	years_at_residence
## 	  2.22%	percent_of_income
## 	  1.56%	dependents
## 	  0.56%	amount
## 
## 
## Time: 0.0 secs
```
La salida muestra algunas de las primeras ramas del árbol de decisiones. Las primeras tres líneas podrían explicarse así:

1. Si el balance de cuenta es desconocido o mayor a 200 DM, se clasifica como "no propenso a impago".
2. Por otra parte, si el balance de cuenta es menor que cero DM, o entre uno y 200 DM.
3. Y el historial de crédito es perfecto o muy bueno, entonces se clasifica como "propenso a impago".

El número entre parentesis indica el número de ejemplos que cumplen el criterio para esa decisión, y el úumero de clasificados incorrectamente por la decisión. Por ejemplo, en la primera línea, *412/50* indica que de 412 ejemplos que llegaron a la decisión, 50 fueron incorrectamente clasificados como no propensos a impago. En otras palabras, 50 clientes impagaron, a expensas de la predicción contraria del modelo.

Algunas veces un árbol produce decisiones con poca lógica. Por ejemplo, ¿porqué un deudor cuyo crédito es muy bueno es propenso a impago, mientras que aquellos cuyo balance de cuenta es desconocido no lo son? Reglas contradictorias como estas ocurren a veces. Esto podría reflejar un patrón real en los datos, o podría ser una anomalía estadística. En cualquier caso, es importante investigar estas decisiones extrañas para ver si la lógica del árbol tiene sentido para usarla en el negocio.

## Paso 4 - Evaluación del rendimiento del modelo
Para aplicar el árbol de decisión al conjunto de prueba, se usa la función *predict()* así:

```r
credito_pred <- predict(credito_modelo, credito_test)
```
Se crea un vector de valores de clase previsto, que se puede comparar con los valores actuales de clase usando la funcion *CrossTable()* del paquete *gmodels*. Estableciendo los parámetros *prop.c* y *prop.r* a FALSE se quitan la columna y la fila de porcentajes de la tabla. El porcentaje restante (*prop.t*) indica la proporción de registros en la celda con respecto al total:

```r
install.packages("gmodels")
```

```
## Installing package into '/home/simon/R/x86_64-pc-linux-gnu-library/3.4'
## (as 'lib' is unspecified)
```

```
## also installing the dependencies 'gtools', 'gdata'
```

```r
library(gmodels)
CrossTable(credito_test$default, credito_pred,
           prop.chisq = FALSE, prop.c = FALSE, prop.r = FALSE,
           dnn = c('impago actual', 'impago predicho'))
```

```
## 
##  
##    Cell Contents
## |-------------------------|
## |                       N |
## |         N / Table Total |
## |-------------------------|
## 
##  
## Total Observations in Table:  100 
## 
##  
##               | impago predicho 
## impago actual |        no |       yes | Row Total | 
## --------------|-----------|-----------|-----------|
##            no |        59 |         8 |        67 | 
##               |     0.590 |     0.080 |           | 
## --------------|-----------|-----------|-----------|
##           yes |        19 |        14 |        33 | 
##               |     0.190 |     0.140 |           | 
## --------------|-----------|-----------|-----------|
##  Column Total |        78 |        22 |       100 | 
## --------------|-----------|-----------|-----------|
## 
## 
```
De 100 registros de prueba de los préstamos, el modelo predijo correctamente  que 59 no impagaron y que 14 lo hicieron, resultando en una precisión del 73 por ciento y una tasa de error del 27 por ciento. Esto es algo peor que el comportamiento en los datos de entrenamiento, pero no es inesperado, ya que el rendimiento del modelo es aún peor en los datos no vistos. Tambien se nota que el modelo solo predijo correctamente 13 de los 33 impagos del conjunto de prueba, o sea un 42 por ciento. Desafortunadamente, este tipo de error es potencialmente muy costoso, ya que el banco pierde dinero con cada incumplimiento. Veamos si podemos mejorar el resultado, con un poco más de trabajo.
