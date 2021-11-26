## Datasets

Folder structure:
 + [FFoCat](https://github.com/ivanDonadello/Food-Categories-Classification)
   + [[Download]](https://scientificnet-my.sharepoint.com/:f:/g/personal/idonadello_unibz_it/EucnGGDEkaJNnd8lY29UilwB1JQfEriHcB6vTn6j0oBrGA?e=0j6ts5)
   + `unzip FFoCat.zip`
 + FFoCat_reduced
```
 folder='FFoCat_reduced' &&
 mkdir -p ${folder}/train ${folder}/valid &&
 cd ${folder} && grep '813\|822\|823\|832' ../FFoCat/food_food_category_map.tsv > food_food_category_map.tsv &&
 cd train && ln -s ../../FFoCat/train/CIRFOOD-{813,822,823,832}* train && cd .. &&
 cd valid && ln -s ../../FFoCat/valid/CIRFOOD-{813,822,823,832}* valid &&
 cd ../..
```
 + FFoCat_tiny (for tests)
   + Copy 8 photos of 4 categories for training and 2 for validation of each.
 + [LTN_ACM_SAC17](https://gitlab.fbk.eu/donadello/LTN_ACM_SAC17/-/tree/master/) (contain pascalpart_dataset)
   + `git clone git@gitlab.fbk.eu:donadello/LTN_ACM_SAC17.git`
 + [OD-MonuMAI](https://github.com/ari-dasci/OD-MonuMAI) 
   + `git clone https://github.com/ari-dasci/OD-MonuMAI`