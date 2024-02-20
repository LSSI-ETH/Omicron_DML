sequence_to_mutation<-function(sequence, position_range=c(331:531), backbone= 'NITNLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGKIADYNYKLPDDFTGCVIAWNSNNLDSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGSTPCNGVEGFNCYFPLQSYGFQPTNGVGYQPYRVVVLSFELLHAPATVCGPKKST'){
  
  seq_split <- str_split(sequence,pattern = "") [[1]]
  backbone_split<-str_split(backbone,pattern = "") [[1]]
  index<-which(seq_split!=backbone_split)
  posi<-position_range[index]
  mutations<-paste0(backbone_split[index],posi,seq_split[index])
  #,collapse = ","
  
  return(mutations)
}

mutation_to_sequence<-function(mutations=c("K417N","E484K","N501Y","L452R","T478K","Q493R"), position_range=c(331:531), backbone= 'NITNLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGKIADYNYKLPDDFTGCVIAWNSNNLDSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGSTPCNGVEGFNCYFPLQSYGFQPTNGVGYQPYRVVVLSFELLHAPATVCGPKKST'){
  
  posi<-str_extract_all(string=mutations,pattern = "(\\d)+",simplify = T)
  posi<-as.numeric(posi)
  mutations<-mutations[!duplicated(posi)]
  
  mut<-str_sub(string = mutations, start = 5, end = 5)
  index<-match(posi,position_range)
  backbone<-str_split(string = backbone,pattern = "")[[1]]
  
  backbone[index]<-mut
  
  backbone<-paste0(backbone,collapse = "")
  return(backbone)
}

remove_unsure<-function(BA1_df1){
  BA1_df1_sure <- subset(BA1_df1, !(Cov22196 ==0.5| X2_7 ==0.5 | ZCB11 ==0.5| ADG20 == 0.5 | BRII198 == 0.5 | A23581 == 0.5 | S2X259 == 0.5 | S2H97 == 0.5)) 
  return(BA1_df1_sure)
}

mutlist_escapeindex<-function(BA1_df1){
  
  mutcol<-str_detect(colnames(BA1_df1),pattern = 'mut') & str_detect(colnames(BA1_df1),pattern = 'BA')
  
  BA1_df1[,mutcol] <- gsub("\\[|\\]|'", "", BA1_df1[,mutcol])
  BA1_df1[,mutcol] <- strsplit(BA1_df1[,mutcol], ", ") 
  BA1_df1$escape_index<-rowSums(BA1_df1[, c('Cov22196','X2_7','ZCB11', 'ADG20', 'BRII198', "A23581", 'S2X259', 'S2H97')]) #here it is still numeric
  return(BA1_df1)
}

add_key_mut<-function(BA1_df1){
  mutcol<-str_detect(colnames(BA1_df1),pattern = 'mut') & str_detect(colnames(BA1_df1),pattern = 'BA')
  count<-c()
  key_mut<-list()
  for(i in 1:nrow(BA1_df1)){
    count[i]<-length(intersect(BA1_df1[,mutcol][[i]],key))
    key_mut[[i]]<- intersect(BA1_df1[,mutcol][[i]],key)
  }
  #add cols
  BA1_df1$count<-count
  BA1_df1$key_mut<-key_mut
  return(BA1_df1)
}




calculate_mut_wt<-function(BA1_df1){
  mut_wt<-list()
  for(i in 1:nrow(BA1_df1)){
    mut_wt[[i]]<- sequence_to_mutation(BA1_df1$seq[i])
  }
  BA1_df1$mut_wt<-mut_wt
  return(BA1_df1)
}

fun<-function(vec1, list){
  out<- any(sapply(list, function(vec) all(vec %in% vec1)))
  return(out)
}

fun1 <- function(vec1, vec2) {
  out<- length(intersect(vec1, vec2))>0
  return(out)
}

fun3<-function(vec1, vec2){
  out <- all(vec1 %in% vec2)
  return(out)
}
fun4<-function(vec1,vec2){
  out <- length(vec1)-sum(vec1 %in% vec2)
  return(out)
}
fun5<-function(vec1,vec2){
  out <- length(vec1)-sum(vec1 %in% vec2) ==1
  return(out)
}

fun6<-function(vec1,vec2){
  out <- vec1[!(vec1 %in% vec2)]
  return(out)
}


top_mut<-function(key_strength, n, type){
  if(type==1){
    key_strength<-subset(key_strength, type %in% c('Observed mutation','Observed mutation position'))
  }
  if(type==2){
    key_strength<-subset(key_strength, type == 'New mutation' | type == 'Observed mutation position')
  }
  if(type==3){
    key_strength<-subset(key_strength, type == 'New mutation')
  }
  key_strength<-key_strength[order(key_strength$strength_adj,decreasing = T),]
  keep_notkey<-key_strength$mut[1:n]
  return(keep_notkey)
}



get_top_mut<-function(key_df, key_strength, type=1, n){
  #key<-c("D339H", "R346T", "L368I", "T376A", "R408S", "D405N", "K444T", "V445P", "S446G", "L452R", "L452Q", "N460K", "F486V", "F486S", "F490S", "R493Q", "S496G", "R346K", "L371F", "L452M", "N417T",
         # "K440N", "A484V", "N417K", "D339G", "L371S", "P373S", "F375S", "N477S","A484E","R498Q","Y501N","H505Y","F486P","N501Y","E484K","K417N","K417T","E484Q","P681R","T478K","Q493R",
         # "A484K","A484Q","S468P","K478T","R346S","V367F","N439K","S494P","Q414K","N450K","N394S","Y449N","F490R","K478R","Y449H","V445A","V362F","V367L","Q414H","K444N","S446V",
         #  "L455F","N477I","N477G","S494L","N417R","K440S","T470N")
  #key_strength<-calculate_strength(key_df)
  #key_strength$type<-ifelse(key_strength$mut %in% key, "Specific mutation in VOC", ifelse(key_strength$posi %in% key_posi,'Positional mutation in VOC', 'Other mutation'))
  #sort(key_strength$strength_adj,decreasing = T)

  # if(type==1){
  #   key_strength<-subset(key_strength, type %in% c('Observed mutation', 'Observed mutation position'))
  # }
  # if(type==2){
  #   key_strength<-subset(key_strength, type == 'New mutation' | type == 'Observed mutation position')
  # }
  # if(type==3){
  #   key_strength<-subset(key_strength, type == 'New mutation')
  # }
  top_key_mut<-top_mut(key_strength, n, type)
  key_df_selected <-key_df[sapply(X=key_df$mut_BA.1, FUN=fun3, vec2 = top_key_mut), ] #152
  #key_df_selected<-add_key_mut(key_df_selected)
  #key_df_selected<-remove_unsure(key_df_selected) #78
  return(key_df_selected)
}


calculate_strength<-function(df, mab_n = 6){
  result <- df %>%
    group_by(escape_index) %>%
    reframe(frequency = table(unlist(mut_BA.1)), mut = names(table(unlist(mut_BA.1))), sum = sum(frequency), len = mean(dis_BA.1), n = n())
  
  result$percentage <- result$frequency/result$sum
  result$mut_escape_value<- (mab_n - result$escape_index) * result$percentage * result$len / result$n
  
  mut_strength<-result %>%
    group_by(mut) %>%
    reframe(strength = sum(mut_escape_value),n = n())
  
  mut_strength$strength_adj<-mut_strength$strength/mut_strength$n
  mut_strength$posi<-as.numeric(str_extract_all(mut_strength$mut, pattern = "(\\d)+", simplify = T))
  mut_strength<-mut_strength[order(mut_strength$posi),]
  
  return(mut_strength)
}



plot_strength <- function(mut_strength, threshold = 0.5e-5, threshold_text = 0.5e-5, color_column = 'type',top = 50) {
  #mut_strength <- subset(mut_strength, type == 'VOC mutation' | strength_adj > threshold)
  mut_strength <- subset(mut_strength, strength_adj > threshold)
  mut_strength<-mut_strength[1:top,]
  mut_strength$mut <- factor(mut_strength$mut, levels = mut_strength$mut[order(mut_strength$posi)])
  custom_colors <- c("Observed mutation" ="#B3593D", 'Observed mutation position' = "#41758A", 'New mutation'='#F4D03F')
  p <- ggplot(mut_strength, aes(x = mut, y = strength_adj, fill = eval(parse(text = color_column)))) +
    geom_bar(stat = "identity") +
    # geom_text_repel(
    #   data =subset(mut_strength, strength_adj > threshold_text),
    #   aes(label = mut),
    #   size = 2,
    #   nudge_y = 0.05e-7,
    #   direction = "y",
    #   segment.color = "black"
    # ) + 
    xlab("Mutations") +
    ylab("Escape score") +
    theme_minimal() +
    theme(
      axis.text.x = element_text(angle = 45, hjust = 1.1, vjust = 1.5),
      axis.text.y = element_text(size = 10),
      panel.grid = element_blank()
    ) +
    scale_fill_manual(values = custom_colors,breaks = names(custom_colors)) +
    labs(fill = "Mutation type")
  
  return(p)
}

#plot function for 2.86
plot_strength_86 <- function(mut_strength,threshold=0.5e-5, color_column = 'type', top=50){
  mut_strength<-  subset(mut_strength,type=='VOC mutation'|strength_adj>=threshold)
  mut_strength<- mut_strength[order(mut_strength$strength_adj,decreasing = T),]
  mut_strength<-mut_strength[1:top,]
  mut_strength$mut<-factor(mut_strength$mut, levels = mut_strength$mut[order(mut_strength$strength_adj,decreasing = T)])
  custom_colors <- c("Specific mutation in BA.2.86" ="#B3593D", 'Mutated site in BA.2.86' = "#41758A",'Other mutation'="#7C7A7A")
  p<-ggplot(mut_strength, aes(x = mut, y = log(strength_adj, 10) + 7 ,fill = eval(parse(text=color_column)))) +
    geom_bar(stat = "identity") +
    #geom_text(data = subset(mut_strength, strength_adj > threshold),
    #         aes(label = mut, y = strength_adj), vjust = -0.5, size = 2) +
    xlab("Mutations") +
    ylab("Adjusted escape score") +
    theme_minimal() +
    #scale_y_log10() + 
    scale_y_continuous(breaks = seq(0, 3, by = 1), minor_breaks = seq(0, 3, by = 0.1)) +
    theme(axis.text.x = element_text(angle = 45, hjust = 1),
          axis.text.y = element_text(size = 10),
          panel.grid = element_blank(),
          legend.position = "bottom") +
    scale_fill_manual(values = custom_colors,breaks = names(custom_colors)) +
    labs(fill = "Mutation type")
    #coord_flip()
  return(p)
}

generate_combinations <- function(vec) {#no use
  n <- length(vec)
  combination_list <- list()
  
  for (i in 1:n) {
    comb <- combn(vec, i)
    for (j in 1:ncol(comb)) {
      combination_list <- c(combination_list, list(comb[, j]))
    }
  }
  
  return(combination_list)
}

find_common_pair<-function(string_list,n ){
  # Create all possible pairs of strings from the list
  string_list<-keep(string_list, ~ length(.) >= n)
  string_pairs <- lapply(string_list, function(x) combn(x, n, simplify = FALSE))
  
  # Flatten the list of pairs
  all_pairs <- unlist(string_pairs, recursive = FALSE)
  
  # Count the occurrences of each pair
  pair_count <- as.data.frame(table(sapply(all_pairs, paste, collapse = ","))) 
  
  # Find the pair with the highest count
  #most_common_pair <- strsplit(names(pair_count)[which.max(pair_count)], ",")
  
  return(pair_count)
}

table_most_escaping_combi<-function(most_escape_combi, table_reference){
  len<-length(most_escape_combi)
  
  range_l<-range(sapply(most_escape_combi,FUN = length))
  if(range_l[1] == 1)  range_l[1]<-2

  for(n in range_l[1]:range_l[2]){
    if(n==range_l[1]){
      table<-find_common_pair(most_escape_combi, n)
    }else
      table<-rbind(table,find_common_pair(most_escape_combi, n))
  }
  table$Freq<-table$Freq/len
  table$baseline<-567476/len
  #something is off here
  table <- table %>%
    inner_join(table_reference, by = "Var1") %>%
    mutate(Freq_ratio = Freq.x / Freq.y) %>%
    select(Var1, Freq_ratio, baseline)
  
  #table <-table[order(table$Freq_ratio,decreasing = T),]
return(table)
}

table_most_escaping_combi_reference<-function(most_escape_combi ){
  len<-length(most_escape_combi)
  
  range_l<-range(sapply(most_escape_combi,FUN = length))
  if(range_l[1] == 1)  range_l[1]<-2
  
  for(n in range_l[1]:range_l[2]){
    if(n==range_l[1]){
      table<-find_common_pair(most_escape_combi, n)
    }else
      table<-rbind(table,find_common_pair(most_escape_combi, n) )
  }
  table$Freq<-table$Freq/len
  
  #table <-table[order(table$Freq,decreasing = T),]
  return(table)
}
