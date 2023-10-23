# Code adapted from the appyter-catalog github repository (https://github.com/MaayanLab/appyter-catalog/tree/main),
# distributed under an Attribution-NonCommercial-ShareAlike 4.0 International license.

# Librairies
import os
import pandas as pd
import numpy as np
import plotly
import plotly.graph_objs as go
from itertools import combinations
from rpy2 import robjects
from rpy2.robjects import r, pandas2ri


# R code
## Here are some useful R functions.
##     Dataframe df: dim(df), rownames(df), colnames(df), df$colname
##     Vector X: X[1:10], sort(X), identical(X, Y), c(2, 23, 4)
##     Print: print(X), impossible to print several elements (ex: print(X, Y))
robjects.r('''limma <- function(rawcount_dataframe, design_dataframe, filter_genes=FALSE, adjust="BH") {
    # Load packages
    suppressMessages(require(limma))
    suppressMessages(require(edgeR))
    # Convert design matrix
    design <- as.matrix(design_dataframe)
    # Create DGEList object
    dge <- DGEList(counts=rawcount_dataframe)
    # Filter genes
    if (filter_genes) {
        keep <- filterByExpr(dge, design)
        dge <- dge[keep,]
    }
    # Calculate normalization factors
    dge <- calcNormFactors(dge)
    # Run VOOM
    v <- voom(dge, design, plot=FALSE)  ## Design added here. Original: v <- voom(dge, plot=FALSE)
    # Fit linear model
    fit <- lmFit(v, design)
    # Make contrast matrix
    cont.matrix <- makeContrasts(de=B-A, levels=design)
    # Fit
    fit2 <- contrasts.fit(fit, cont.matrix)
    # Run DE
    fit2 <- eBayes(fit2)
    # Get results
    limma_dataframe <- topTable(fit2, adjust=adjust, number=nrow(rawcount_dataframe))
    # Return
    results <- list("limma_dataframe"= limma_dataframe, "rownames"=rownames(limma_dataframe), "colnames"=colnames(limma_dataframe))
    return (results)
}
''')


robjects.r('''deseq2 <- function(rawcount_dataframe, g1, g2) {
    # Parameters
    #     rawcount_dataframe -- R dataframe containing raw counts. Shape (n_gene, n_sample).
    #     g1 -- vector containing the names of the samples belonging to the "Control" group.
    #     g2 -- vector containing the names of the samples belonging to the "Condition" group.
    
    # Load packages
    suppressMessages(require(DESeq2))

    # Prepare data
    colData <- as.data.frame(c(rep(c("Control"),length(g1)),rep(c("Condition"),length(g2))))  # "Control" ... "Control" "Condition" ... "Condition"
    rownames(colData) <- c(g1,g2)  # name of the rows (samples of group g1 followed by samples of group g2)
    colnames(colData) <- c("group")  # name of the column
    colData$group = relevel(as.factor(colData$group), "Control")  # "Control"is the first level, "Condition" is the second.
    # colData:  GTEX-S32W-2026-SM-4AD6E  Condition

    rawcount_dataframe <- rawcount_dataframe[, rownames(colData)]  # the order of rawcount_dataframe columns must be the same as the rows of colData.
    
    # Method
    dds <- DESeqDataSetFromMatrix(countData = round(rawcount_dataframe), colData = colData, design=~(group))  # rawcount_dataframe must contain integer values
    dds <- DESeq(dds)  
    res <- results(dds)  # log2 fold change (MLE): condition a vs b: log2(a/b).
    res[which(is.na(res$padj)),] <- 1  # padj: adjusted p value. NA values replaced by 1.
    res <- as.data.frame(res)
    results <- list("DESeq_dataframe"= res, "rownames"=rownames(res), "colnames"=colnames(res))

    return(results)
}
''')




# Python function
def get_signatures(classes, expression, phenotype, method):
    """Parameters:
        classes   --  List containing the names of the studied groups. Ex : ['Tumor', 'Normal'].
        expression  --  Pd dataframe whose rows are genes, columns are samples and values show the gene expression level.
        phenotype --  Pd dataframe whose rows are and columns are.
        method  --  "limma" or "DESeq2".
    """
    pandas2ri.activate()  # activation needed to convert pandas objects to R.
    signatures = dict()

    for cls1, cls2 in combinations(classes, 2):    # 2-length combinations of elements in classes
        cls1_sample_ids = phenotype.loc[phenotype==cls1].index.tolist()  # Control
        cls2_sample_ids = phenotype.loc[phenotype==cls2].index.tolist()  # Case
        signature_label = " vs. ".join([cls1, cls2])

        if method == "limma":
            limma = robjects.r['limma']
            design_dataframe = pd.DataFrame([{'index': x, 'A': int(x in cls1_sample_ids), 'B': int(x in cls2_sample_ids)} for x in expression.columns]).set_index('index')
            a = limma(pandas2ri.conversion.py2rpy(expression), pandas2ri.conversion.py2rpy(design_dataframe), filter_genes=False)
            limma_results = pandas2ri.conversion.rpy2py(a)
            signature = pd.DataFrame(limma_results[0]).T
            signature.index = limma_results[1]
            signature.columns = limma_results[2]
            signature = signature.sort_values("t", ascending=False)

        elif method == "DESeq2":
            DESeq2 = robjects.r['deseq2']
            a = DESeq2(pandas2ri.conversion.py2rpy(expression), pandas2ri.conversion.py2rpy(cls1_sample_ids), pandas2ri.conversion.py2rpy(cls2_sample_ids))
            DESeq2_results = pandas2ri.conversion.rpy2py(a)
            signature = pd.DataFrame(DESeq2_results[0]).T
            signature.index = DESeq2_results[1]
            signature.columns = DESeq2_results[2]
            signature = signature.sort_values("log2FoldChange", ascending=False)

        signatures[signature_label] = signature

    return signatures



# Plots
def run_volcano(signature, signature_label, pvalue_threshold, logfc_threshold, plot_type):
    color = []
    text = []
    print("Run volcano")
    for index, rowData in signature.iterrows():
        if "AveExpr" in rowData.index: # limma
            expr_colname = "AveExpr"
            pval_colname = "P.Value"
            logfc_colname = "logFC"
        elif "logCPM" in rowData.index: #edgeR
            expr_colname = "logCPM"
            pval_colname = "PValue"
            logfc_colname = "logFC"
        elif "baseMean" in rowData.index: #DESeq2
            expr_colname = "baseMean"
            pval_colname = "padj"  # "pvalue"
            logfc_colname = "log2FoldChange"
        # Text
        text.append('<b>'+index+'</b><br>Avg Expression = '+str(round(rowData[expr_colname], ndigits=2))+'<br>logFC = '+str(round(rowData[logfc_colname], ndigits=2))+'<br>p = '+'{:.2e}'.format(rowData[pval_colname])+'<br>FDR = '+'{:.2e}'.format(rowData[pval_colname]))

        # Color
        if rowData[pval_colname] < pvalue_threshold:
            if rowData[logfc_colname] < -logfc_threshold:
                color.append('blue')
            elif rowData[logfc_colname] > logfc_threshold:
                color.append('red')
            else:
                color.append('black')

        else:
            color.append('black')
    volcano_plot_results = {'x': signature[logfc_colname], 'y': -np.log10(signature[pval_colname]), 'text':text, 'color': color, 'signature_label': signature_label, 'plot_type': plot_type}
    return volcano_plot_results


def plot_2D_scatter(x, y, text='', title='', xlab='', ylab='', hoverinfo='text', color='black', colorscale='Blues', size=8, showscale=False, symmetric_x=False, symmetric_y=False, pad=0.5, hline=False, vline=False, return_trace=False, labels=False, plot_type='interactive', de_type='ma', plot_name="2d_plot.png"):
    range_x = [-max(abs(x))-pad, max(abs(x))+pad]if symmetric_x else None
    range_y = [-max(abs(y))-pad, max(abs(y))+pad]if symmetric_y else None
    trace = go.Scattergl(x=x, y=y, mode='markers', text=text, hoverinfo=hoverinfo, marker={'color': color, 'colorscale': colorscale, 'showscale': showscale, 'size': size})
    if return_trace:
        return trace
    else:
        if de_type == 'ma':
            annotations = [
                {'x': 1, 'y': 0.1, 'text':'<span style="color: blue; font-size: 10pt; font-weight: 600;">Down-regulated in '+labels[-1]+'</span>', 'showarrow': False, 'xref': 'paper', 'yref': 'paper', 'xanchor': 'right', 'yanchor': 'top'},
                {'x': 1, 'y': 0.9, 'text':'<span style="color: red; font-size: 10pt; font-weight: 600;">Up-regulated in '+labels[-1]+'</span>', 'showarrow': False, 'xref': 'paper', 'yref': 'paper', 'xanchor': 'right', 'yanchor': 'bottom'}
            ] if labels else []
        elif de_type == 'volcano':
            annotations = [
                {'x': 0.25, 'y': 1.07, 'text':'<span style="color: blue; font-size: 10pt; font-weight: 600;">Down-regulated in '+labels[-1]+'</span>', 'showarrow': False, 'xref': 'paper', 'yref': 'paper', 'xanchor': 'center'},
                {'x': 0.75, 'y': 1.07, 'text':'<span style="color: red; font-size: 10pt; font-weight: 600;">Up-regulated in '+labels[-1]+'</span>', 'showarrow': False, 'xref': 'paper', 'yref': 'paper', 'xanchor': 'center'}
            ] if labels else []
        layout = go.Layout(title=title, xaxis={'title': xlab, 'range': range_x}, yaxis={'title': ylab, 'range': range_y}, hovermode='closest', annotations=annotations)
        fig = go.Figure(data=[trace], layout=layout)
    
    if plot_type=='interactive':
        fig.show()
    # else:
    #    fig.show(renderer="png")
    fig.write_image(plot_name)


def plot_volcano(volcano_plot_results, save_path):
    spacer = ' '*50
    plot_name = os.path.join(save_path, "volcano_plot_{}.png".format(volcano_plot_results['signature_label']))
    plot_2D_scatter(
        x=volcano_plot_results['x'],
        y=volcano_plot_results['y'],
        text=volcano_plot_results['text'],
        color=volcano_plot_results['color'],
        symmetric_x=True,
        xlab='log2FC',
        ylab='-log10P',
        title='<b>{volcano_plot_results[signature_label]} Signature | Volcano Plot</b>'.format(**locals()),
        labels=volcano_plot_results['signature_label'].split(' vs. '),
        plot_type=volcano_plot_results['plot_type'],
        de_type='volcano',
        plot_name=plot_name
    )        
    return plot_name
