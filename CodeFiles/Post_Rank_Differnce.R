
#Getting the dataset for calculating difference in rank

diff_rank=reddit1900[,c("Rank","id","Repeated")]
#Adding this id column to reoirder the dataframe later to merge with the larger reddit dataset 
diff_rank$Seq=seq.int(nrow(diff_rank))
#Ordering the dataset by the ids to ensure that same ids are placed next to each other 
diff_rank=diff_rank[order(diff_rank$id),]
diff=c()

for ( i in 1:nrow(diff_rank)){
  
  if(diff_rank$Repeated[i]==0 || diff_rank$Repeated[i]==1){
     diff[i]=0
  }else if (diff_rank$Repeated[i]==2 || diff_rank$Repeated[i]==3 || diff_rank$Repeated[i]==4|| diff_rank$Repeated[i]==5 ){
       current_rank=diff_rank$Rank[i]
       previous_rank=diff_rank$Rank[i-1]
       diff[i]=(previous_rank-current_rank)
       
     }
}
diff=as.data.frame(diff)
diff_rank_final=as.data.frame(cbind(diff_rank,diff))
diff_rank_final=diff_rank_final[order(diff_rank_final$Seq),]

#Adding it to the final data
reddit1900<-as.data.frame(cbind(reddit1900,diff_rank_final$diff))


#Getting the dataset for calculating difference in scores to ensure that repeated stories do not 
diff_scores=reddit1900[,c("score","id","Repeated")]

#Adding this id column to reoirder the dataframe later to merge with the larger reddit dataset 
diff_scores$Seq=seq.int(nrow(diff_scores))
#Ordering the dataset by the ids to ensure that same ids are placed next to each other 
diff_scores=diff_scores[order(diff_scores$id),]
diff=c()

for ( i in 1:nrow(diff_scores)){
  
  if(diff_scores$Repeated[i]==0 || diff_scores$Repeated[i]==1){
    diff[i]=diff_scores$score[i]
  }else if (diff_scores$Repeated[i]==2 || diff_scores$Repeated[i]==3 || diff_scores$Repeated[i]==4|| diff_scores$Repeated[i]==5 ){
    current_score=diff_scores$score[i]
    previous_score=diff_scores$score[i-1]
    diff[i]=( current_score-previous_score)
    
  }
}

diff=as.data.frame(diff)
diff_score_final=as.data.frame(cbind(diff_scores,diff))
diff_score_final=diff_score_final[order(diff_score_final$Seq),]

write.csv(diff_score_final,file="diff_scores.csv")

scores=read.csv(file.choose())
summary(scores$LR)
scores$levelofengagement_likes=ifelse(scores$LR<0.005292,"Low",ifelse(scores$LR>0.005292 & scores$LR<0.054366,"Medium","High" ))
summary(factor(scores$levelofengagement_likes))
write.csv(scores,"Scores_TransformedV2.csv")



#Getting the dataset for calculating difference in number of comments to ensure that repeated stories do not 
diff_comms=reddit1900[,c("comms_num","id","Repeated")]

#Adding this id column to reoirder the dataframe later to merge with the larger reddit dataset 
diff_comms$Seq=seq.int(nrow(diff_comms))
#Ordering the dataset by the ids to ensure that same ids are placed next to each other 
diff_comms=diff_comms[order(diff_comms$id),]
diff=c()

for ( i in 1:nrow(diff_comms)){
  
  if(diff_comms$Repeated[i]==0 || diff_comms$Repeated[i]==1){
    diff[i]=diff_comms$comms_num[i]
  }else if (diff_comms$Repeated[i]==2 || diff_comms$Repeated[i]==3 || diff_comms$Repeated[i]==4|| diff_comms$Repeated[i]==5 ){
    current_comms=diff_comms$comms_num[i]
    previous_comms=diff_comms$comms_num[i-1]
    diff[i]=( current_comms-previous_comms)
    
  }
}

diff=as.data.frame(diff)
diff_comms_final=as.data.frame(cbind(diff_comms,diff))
diff_comms_final=diff_comms_final[order(diff_comms_final$Seq),]

write.csv(diff_comms_final,file="diff_comms.csv")

comms=read.csv(file.choose())
summary(comms$CR)
comms$levelofengagement_comms=ifelse(comms$CR<0.001324,"Low",ifelse(comms$CR>0.001324 & comms$CR<0.020085,"Medium","High" ))
summary(factor(comms$levelofengagement_comms))
write.csv(comms,file="comments_transformed.csv")

#Getting the dataset for calculating difference in number of comments to ensure that repeated stories do not 
diff_share=reddit1900[,c("cross_posts","id","Repeated")]

#Adding this id column to reoirder the dataframe later to merge with the larger reddit dataset 
diff_share$Seq=seq.int(nrow(diff_share))
#Ordering the dataset by the ids to ensure that same ids are placed next to each other 
diff_share=diff_share[order(diff_share$id),]
diff=c()

for ( i in 1:nrow(diff_share)){
  
  if(diff_share$Repeated[i]==0 || diff_share$Repeated[i]==1){
    diff[i]=diff_share$cross_posts[i]
  }else if (diff_share$Repeated[i]==2 || diff_share$Repeated[i]==3 || diff_share$Repeated[i]==4|| diff_share$Repeated[i]==5 ){
    current_share=diff_share$cross_posts[i]
    previous_share=diff_share$cross_posts[i-1]
    diff[i]=( current_share-previous_share)
    
  }
}

diff=as.data.frame(diff)
diff_share_final=as.data.frame(cbind(diff_share,diff))
diff_comms_final=diff_comms_final[order(diff_comms_final$Seq),]

write.csv(diff_share_final,file="diff_share.csv")
s=read.csv(file.choose())
summary(s$SR)
s$levelofengagement_sharing=ifelse(s$SR==0,"Low",ifelse(s$SR>0 & s$SR<4.000e-05,"Medium","High" ))
summary(factor(s$levelofengagement_sharing))
write.csv(s,file="sharing_transformed.csv")
