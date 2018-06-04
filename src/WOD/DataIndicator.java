package WOD;

import java.util.ArrayList;
import java.util.Hashtable;
import java.util.List;
import java.util.Map;

import weka.attributeSelection.InfoGainAttributeEval;
import weka.attributeSelection.Ranker;
import weka.core.Instance;
import weka.core.Instances;

public class DataIndicator {
	
	
	
	public static double calcSeperability(Instances instances, int[] valueFrequency, int[] firstValueIndex, List<String> listOfClass) {

		int nFeature = instances.numAttributes()-1;
		int nObject = instances.numInstances();
		double maxAuc = 0.0;
		List<Double> aucList = new ArrayList<>();
		
		
		for(int i = 0; i < nFeature; i++) {
			
			double[] objectScore = new double[nObject];
			for(int j = 0; j < nObject; j++) {
				Instance instance = instances.instance(j);
				double value = instance.value(i);

				int valueIndex = firstValueIndex[i] + (int) value;
				double score = 1 - ((double)valueFrequency[valueIndex] / (double)nObject);
				objectScore[j] = score;
			}
			
			Hashtable<Integer, Double> objectScoreTable = OutliernessEvaluator.GenerateObjectScoreMap(objectScore);		
	    	Evaluation evaluation = new Evaluation("outlier");
	    	double auc = evaluation.computeAUCAccordingtoOutlierRanking(listOfClass, 
	    			evaluation.rankInstancesBasedOutlierScores(objectScoreTable));
	    	if(auc > maxAuc) {
	    		maxAuc = auc;
	    	}
	    	aucList.add(auc);
	    	
		}

    	return maxAuc;		
	}
	
	
	
	
	
	public static double calcNoisyRate(Instances instances, int[] valueFrequency, int[] firstValueIndex, List<String> listOfClass) {

		int nFeature = instances.numAttributes()-1;
		int nObject = instances.numInstances();
		int count = 0;
//		List<Double> aucList = new ArrayList<>();
		
		
		for(int i = 0; i < nFeature; i++) {
			
			double[] objectScore = new double[nObject];
			for(int j = 0; j < nObject; j++) {
				Instance instance = instances.instance(j);
				double value = instance.value(i);

				int valueIndex = firstValueIndex[i] + (int) value;
				double score = 1 - ((double)valueFrequency[valueIndex] / (double)nObject);
				objectScore[j] = score;
			}
			
			Hashtable<Integer, Double> objectScoreTable = OutliernessEvaluator.GenerateObjectScoreMap(objectScore);		
	    	Evaluation evaluation = new Evaluation("outlier");
	    	double auc = evaluation.computeAUCAccordingtoOutlierRanking(listOfClass, 
	    			evaluation.rankInstancesBasedOutlierScores(objectScoreTable));
	    	if(auc <= 0.5) {
	    		count++;
	    	}	    	
		}
		double noisyRate = (double)count / (double)nFeature;

    	return noisyRate;		
	}
	
	
	
	
	public static double calcIG(Instances instances) throws Exception {
		instances.setClassIndex(instances.numAttributes()-1); 
        Ranker rank = new Ranker();  
        
        InfoGainAttributeEval eval = new InfoGainAttributeEval();  
        eval.buildEvaluator(instances);  
//        int[] attrIndex = rank.search(eval, instances);    
        
      
        //infoGain value
        double infoGainSum = 0.0;
        double infoGainAvg = 0.0;
        double maxInfoGain = 0.0;
        for(int i = 0; i < instances.numAttributes()-1; i++){
        	double infogain = eval.evaluateAttribute(i);
        	infoGainSum += infogain;
        	if(infogain > maxInfoGain) {
        		maxInfoGain = infogain;
        	}
        	//infoGainMap.put(instances.attribute(attrIndex[i]).name(), eval.evaluateAttribute(attrIndex[i]));
        }
        
        infoGainAvg = infoGainSum / (double) (instances.numAttributes()-1);
        
        return maxInfoGain;
	}
	
	
	
	
	
	
	public static double calcCorrelationStrength(List<Integer> ValueList, double[] conditionalPossibilityWithLabel) {
		double avg = 0.0;
		double tmpSum = 0.0;
		int size = ValueList.size();
		for(int i = 0; i < size; i++) {
			int index = ValueList.get(i);
			tmpSum += conditionalPossibilityWithLabel[index];
		}
		avg = tmpSum / (double) size;
		
		return avg;
	}
	
	
	
	public static double calcCorrelationStrength2(List<Integer> ValueList, double[] conditionalPossibilityWithLabel) {
		double sum = 0.0;
		int size = ValueList.size();
		for(int i = 0; i < size; i++) {
			int index = ValueList.get(i);
			sum += conditionalPossibilityWithLabel[index];
		}
		
		return sum;
	}





	public static double calcCorrelationStrength_weight(double[] conditionalPossibilityWithLabel,
			int[] clusterInfo,
			double[] clusterWeight,
			Map<Integer, Integer> clusterSizeMap) {

		
		int nCluster = clusterWeight.length;
		double[] clusterACS = new double[nCluster];
	
		
		for(int i = 0; i < conditionalPossibilityWithLabel.length; i++) {
			int clusterIndex = clusterInfo[i];
			clusterACS[clusterIndex] += conditionalPossibilityWithLabel[i];
		}
		
		
		for(int i = 0; i < nCluster; i++) {
			int clusterSize = clusterSizeMap.get(i);
			clusterACS[i] = clusterACS[i] / clusterSize;
		}
		
		
		double finalACS = 0.0;
		for(int i = 0; i < clusterACS.length; i++) {
			finalACS += clusterACS[i] * clusterWeight[i];
		}
	
		return finalACS;
	}

	
	public static double calcCorrelationStrength_weight2(double[] conditionalPossibilityWithLabel,
			int[] clusterInfo,
			double[] clusterWeight,
			Map<Integer, Integer> clusterSizeMap) {

		
		int nValue = conditionalPossibilityWithLabel.length;
		double weightedAcs = 0.0;
	
		
		for(int i = 0; i < conditionalPossibilityWithLabel.length; i++) {
			int clusterIndex = clusterInfo[i];
			double clusterweight = clusterWeight[clusterIndex];	
			weightedAcs += conditionalPossibilityWithLabel[i] * clusterweight;
		}
		
		weightedAcs = weightedAcs / nValue;
	
		return weightedAcs;
	}
	
	
	

}
