package WOD;


import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Hashtable;
import java.util.List;
import Louvain.ModularityOptimizer;


/**
 * 
 * @author Hongzuo Xu
 * @version 2018.3 - DEXA2018
 * 
 */
public class ODutil {

	public static void main(String[] args) throws Exception {
		//path of single data set file of path of data set folder
		
		String path = args[0];
		//String path = "E:\\data\\MDODarff\\finalData\\";
		
		if(path.endsWith(".arff")) {
			runWOD(path);
		}else {
			List<String> datasetList = buildDataSetsPathList(path);
			for(int i = 0; i <datasetList.size(); i++) {
				runWOD(datasetList.get(i));
			}
		}
	}
	
	
	
	public static void runDataIndicator(String path) throws Exception {
		NetworkConstructor networkConstructor = new NetworkConstructor();
		networkConstructor.calCPWithLabel();
		
    	String name = getDatasetName(path);
    	System.out.print(name + ",");
    	networkConstructor.dataPrepareFromArff(path);
    	double separability = DataIndicator.calcSeperability(networkConstructor.getInstances(), 
    			networkConstructor.getValueFrequency(),
    			networkConstructor.getFirstValueIndex(), 
    			networkConstructor.getListOfClass());
    	
    	double noisyRate = DataIndicator.calcNoisyRate(networkConstructor.getInstances(), 
    			networkConstructor.getValueFrequency(),
    			networkConstructor.getFirstValueIndex(), 
    			networkConstructor.getListOfClass());
    	

    	System.out.format("sep,%.4f," , separability);
    	System.out.format("noisyRate,%.4f," , noisyRate);	
	}
	


	
	
	public static void runWOD(String path) throws IOException {
		NetworkConstructor networkConstructor = new NetworkConstructor();
		
    	String name = getDatasetName(path);
    	System.out.print(name + ",");
		
		long beginTime = System.currentTimeMillis();
		networkConstructor.dataPrepareFromArff(path);
		networkConstructor.calCPWithLabel();
		List<String> networkList =  networkConstructor.getNetworkList(0.0);	
		
		ModularityOptimizer modularityOptimizer = new ModularityOptimizer();
		int[] clusterInfo = modularityOptimizer.runLouvain(networkList);

		
		
		List<Integer> fullValueList = new ArrayList<>();
		for(int i = 0; i < networkConstructor.getNValues(); i++) {
			fullValueList.add(i);
		}	

		//identify normal-value cluster
		double[] initValueOutlierness = OutliernessEvaluator.calInitValueOutlierness(
				networkConstructor.getFirstValueIndex(), 
				networkConstructor.getValueFrequency());
		double[] clusterOutlierness1 = OutliernessEvaluator.calcClusterOutlierness(clusterInfo, 
				modularityOptimizer.getNClusters(), 
				initValueOutlierness);
		int normalClusterIndex = OutliernessEvaluator.getNormalClusterId(clusterOutlierness1);	
				
		// identify noisy-value cluster 
		int threhold = OutliernessEvaluator.calcThrehold_np(modularityOptimizer.getNClusters(), 
				networkConstructor.getNValues(), 
				modularityOptimizer.getClusterSizeMap(), 
				normalClusterIndex);	
			
		List<Integer> remainingValueList = OutliernessEvaluator.getRemainingClusterValueList(clusterInfo, normalClusterIndex, 
				modularityOptimizer.getNClusters(),
				modularityOptimizer.getClusterSizeMap(),
				threhold);
			
		//calc value weight
		double[] valueWeight = OutliernessEvaluator.valueWeightScoring(
				networkConstructor.getSimilarityMatrix(), 
				initValueOutlierness, 
				remainingValueList);
				
		//calc cluster weight
		double[] clusterRawWeight = OutliernessEvaluator.calClustersWeight(
				networkConstructor.getSimilarityMatrix(), 
				clusterInfo, 
				modularityOptimizer.getNClusters(),
				valueWeight);
		double[] clusterWeight = OutliernessEvaluator.getStandardClusterWeight(
				clusterRawWeight, 
				modularityOptimizer.getNClusters(), 
				normalClusterIndex, 
				modularityOptimizer.getClusterSizeMap(), 
				threhold);
	
		
		double[] finalValueScore = OutliernessEvaluator.weightedValueCouplingLearning(
				networkConstructor.getConditionalPossibility(),
				clusterWeight,
				valueWeight, 
				fullValueList, 
				clusterInfo);

		double[] objectScore = OutliernessEvaluator.objectOutliernessScoreing(finalValueScore, 
				networkConstructor.getInstances(),
				networkConstructor.getFirstValueIndex());
		
		
		
        long endTime = System.currentTimeMillis(); 
        
        int usedValueNum = remainingValueList.size();
      
        
		Hashtable<Integer, Double> objectScoreTable = OutliernessEvaluator.GenerateObjectScoreMap(objectScore);		
    	Evaluation evaluation = new Evaluation("outlier");
    	double auc = evaluation.computeAUCAccordingtoOutlierRanking(networkConstructor.getListOfClass(), 
    			evaluation.rankInstancesBasedOutlierScores(objectScoreTable));
        	

    	System.out.print(networkConstructor.getNFeatures() + ",");
    	System.out.print(networkConstructor.getNValues() + ",");
    	System.out.print(networkConstructor.getNObjects() + ",");
    	System.out.format("maxMD,%.4f,", modularityOptimizer.getMaxModularity());
    	System.out.print("usedValueNum," +  usedValueNum + ",");

    	System.out.format("auc,%.4f" , auc);
    	System.out.print(",");
    	
    	
    	System.out.format("%.4fs%n", (endTime - beginTime) / 1000.0);    
    }	
	
	
	


	
	public static String getDatasetName(String path) {
		String name = "";		
		//windows
		if(path.contains("\\")) {
			String[] splitedString = path.split("\\\\");
			name = splitedString[splitedString.length-1].split("\\.")[0];
		}else {
			//linux
			String[] splitedString = path.split("/");
			name = splitedString[splitedString.length-1].split("\\.")[0];
		}	
		return name;
	}
	
	
	
	
    /**
     * to store the file names contained in a folder
     * @param dataSetFilesPath the path of the folder
     */
    public static List<String> buildDataSetsPathList(String dataSetFilesPath)
    {
        File filePath = new File(dataSetFilesPath);
        String[] fileNameList =  filePath.list();
        int dataSetFileCount = 0;
        for (int count=0;count < fileNameList.length;count++)
        {
            // System.out.println(fileNameList[count]);
            if (fileNameList[count].toLowerCase().endsWith(".arff"))
            {
                dataSetFileCount = dataSetFileCount +1;
            }
        }
        List<String> dataSetFullNameList = new ArrayList<>();

        dataSetFileCount = 0;
        for (int count =0; count < fileNameList.length; count++)
        {
            if (fileNameList[count].toLowerCase().endsWith(".arff"))
            {
                if(dataSetFilesPath.contains("\\")) {
                    dataSetFullNameList.add(dataSetFilesPath+"\\"+fileNameList[count]);
                }else {
                    dataSetFullNameList.add(dataSetFilesPath+"/"+fileNameList[count]);

				}

                dataSetFileCount = dataSetFileCount +1;
            }
        }       
        return dataSetFullNameList;
    }    
    
}
