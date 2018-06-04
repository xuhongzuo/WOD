package WOD;

import java.util.ArrayList;
import java.util.Hashtable;
import java.util.List;
import java.util.Map;
import weka.core.Instance;
import weka.core.Instances;


/**
 * 
 * @author Hongzuo Xu
 *
 */
public class OutliernessEvaluator {	
	public static double[] valueWeightScoring(double[][] similarityMatrix, 
			double[] lastValueOutlierness, List<Integer> selectedValueList) {
		int nValue = similarityMatrix.length;
		double[] valueOutlierness = new double[nValue];
		
		double[] columnSum = new double[nValue];
		for(int i = 0; i < nValue; i++) {
			double sum = 0.0;
			for(int j = 0; j < nValue; j++) {
				sum += similarityMatrix[j][i];
			}
			columnSum[i] = sum;
		}	
			
		double[][] normalizedMatrix = new double[nValue][nValue];
		for(int i = 0; i < nValue; i++) {
			for(int j = 0; j < nValue; j++) {
				normalizedMatrix[i][j] = similarityMatrix[i][j] / columnSum[j];
			}
		}
		
		//score each value 
		for(int i = 0; i < nValue; i++) {
			double score = 0.0;
			for(int j = 0; j < selectedValueList.size(); j++) {
				int index = selectedValueList.get(j);
				score += normalizedMatrix[i][index] * lastValueOutlierness[index];
			}
			valueOutlierness[i] = score;
		}
				
		return valueOutlierness;
	}	
	
	
	
	
	
	
	public static double[] weightedValueCouplingLearning(double[][] conditionalPossibility,
			double[] clusterWeight, 
			double[] lastValueOutlierness, 
			List<Integer> selectedValueList, 
			int[] clusterInfo) {
		
		int nValue = conditionalPossibility.length;
		double[] valueOutlierness = new double[nValue];

		
		//column-normalized 
		double[] columnSum = new double[nValue];
		for(int i = 0; i < nValue; i++) {
			double sum = 0.0;
			for(int j = 0; j < nValue; j++) {
				sum += conditionalPossibility[j][i];
			}
			columnSum[i] = sum;
		}	
				
		double[][] normalizedMatrix = new double[nValue][nValue];
		for(int i = 0; i < nValue; i++) {
			for(int j = 0; j < nValue; j++) {
				normalizedMatrix[i][j] = conditionalPossibility[i][j] / columnSum[j];
			}
		}
		
		//score each value 
		for(int i = 0; i < nValue; i++) {
			double score = 0.0;
			for(int j = 0; j < selectedValueList.size(); j++) {
				int index = selectedValueList.get(j);
				int clusterIndex = clusterInfo[index];
				double weight = clusterWeight[clusterIndex];
				score += normalizedMatrix[i][index] * lastValueOutlierness[index] * weight;
				//score += normalizedMatrix[i][index] * weightedValueOutlierness[j];
			}
			valueOutlierness[i] = score;
		}

				
		return valueOutlierness;

	}
	
	
	
	public static double[] objectOutliernessScoreing(double[] valueOutlierness, Instances instances, int[] firstValueIndex) {
		int nObject = instances.numInstances();
		int nFeatures = instances.numAttributes() - 1;

		//calculate weight of each feature
		double[] relevances = new double[nFeatures];
		double relevanceSum = 0.0;
		for(int i = 0; i < nFeatures; i++) {
			double relevance = 0.0;
			for(int j = firstValueIndex[i]; j < firstValueIndex[i+1]; j++) {
				relevance += valueOutlierness[j];
			}
			relevances[i] = relevance;
			relevanceSum += relevance;
		}
		
		double[] weight = new double[nFeatures];
		for(int i = 0; i < nFeatures; i++) {
			weight[i] = relevances[i] / relevanceSum;			
		}
		
		double[] objectOutlierness = new double[nObject];
		for(int i = 0; i < nObject; i++) {
			double score = 0.0;
			Instance instance = instances.instance(i);
			for(int j = 0; j < nFeatures; j++) {
				double value = instance.value(j);
				double featureWeight = weight[j];
				int valueIndex = firstValueIndex[j] + (int) value;
				score += valueOutlierness[valueIndex] * featureWeight;
			}
			objectOutlierness[i] = score;
		}
		
		
		return objectOutlierness;
	}
	
	
	
	public static Hashtable<Integer, Double> GenerateObjectScoreMap(double[] objectScore){
		Hashtable<Integer, Double> objectScoreTable = new Hashtable<>();
		for(int i = 0; i < objectScore.length; i++) {
			objectScoreTable.put(i, objectScore[i]);
		}		
		return objectScoreTable;		
	}
	
	
	
	
	

	/**
	 * calculate outlierness of each cluster based on init outlierness of each value
	 * @param clusterInfo 
	 * @param nCluster
	 * @param valueFrequency
	 * @return double[] index is clusterId, content is the sum of value/node frequency
	 */
	public static double[] calcClusterOutlierness(int[] clusterInfo, int nCluster, double[] valueOutlierness) {
		double[] clusterOutlierness = new double[nCluster];
		int[] clusterSize = new int[nCluster];
		for(int i = 0; i < clusterInfo.length; i++) {
			double outlierness = valueOutlierness[i];
			int clusterId = clusterInfo[i];
			clusterOutlierness[clusterId] += outlierness;
			clusterSize[clusterId]++;
		}		
		
		double tmpSum = 0.0;
		for(int i = 0; i < nCluster; i++) {
			clusterOutlierness[i] = clusterOutlierness[i] / (double) clusterSize[i];
			tmpSum += clusterOutlierness[i];
		}
		
		for(int i = 0; i < nCluster; i++) {
			clusterOutlierness[i] = clusterOutlierness[i] / tmpSum;
		}
		
		
		return clusterOutlierness;		
	}
	

	
	
	
	/**
	 * get selected value subset - the most normal cluster
	 * @param clusterInfo 
	 * @param nCluster
	 * @param valueFrequency
	 * @return double[] index is clusterId, content is the sum of value/node frequency
	 */
	public static int getNormalClusterId(double[] clusterOutlierness) {
		double minOutlierness = 1.0;
		int index = -1;
		for(int i = 0; i < clusterOutlierness.length; i++) {
			if(clusterOutlierness[i] < minOutlierness) {
				index = i;
				minOutlierness = clusterOutlierness[i];
			}
		}
		return index;		
	}
	

	public static List<Integer> getRemainingClusterValueList(int[] clusterInfo, 
			int normalIndex, int nCluster, Map<Integer, Integer> clusterSizeMap, 
			int threhold) {
		List<Integer> filterClusterList = new ArrayList<>();
		filterClusterList.add(normalIndex);
		
		for(int i = 0; i < nCluster; i++) {
			int size = clusterSizeMap.get(i);
			if(size < threhold) {
				filterClusterList.add(i);
			}
		}
			
		List<Integer> list = new ArrayList<>();
		for(int i = 0; i < clusterInfo.length; i++) {
			int clusterIndex = clusterInfo[i];
			if(!filterClusterList.contains(clusterIndex)) {
				list.add(i);
			}
		}
		return list;
	}
	
	
	public static List<Integer> getRemainingFeatureList(List<Integer> remainingValueList, int[] firstValueIndex) {
		List<Integer> remainingFeatureList = new ArrayList<>();
	
		for(int i = 0; i < remainingValueList.size(); i++) {
			int valueIndex1 = remainingValueList.get(i);
			int featureIndex1 = 0;
			for(featureIndex1 = 0; featureIndex1 < firstValueIndex.length-1; featureIndex1++) {
				int featureValueIndex = firstValueIndex[featureIndex1];

				if(valueIndex1 >= featureValueIndex) {
					continue;
				}else {
					break;
				}		
			}
			featureIndex1 = featureIndex1-1;
			//int localValueIndex1 = valueIndex1 - firstValueIndex[featureIndex1];
			if(!remainingFeatureList.contains(featureIndex1)) {
				remainingFeatureList.add(featureIndex1);
			}
			
		}
		
		return remainingFeatureList;
	}
	
	

	
	public static List<Integer> getValueListByCluster(int[] clusterInfo, int index){
		List<Integer> list = new ArrayList<>();
		for(int i = 0; i < clusterInfo.length; i++) {
			if(clusterInfo[i] == index) {
				list.add(i);
			}
		}
		return list;
	}
	
	

	
	
	/**
	 * use function to calculate the initial outlierness of each value
	 * @param firstValueIndex
	 * @param valueFrequency
	 * @return
	 */
	public static double[] calInitValueOutlierness(int[] firstValueIndex, int[] valueFrequency) {
		int nValue = valueFrequency.length;
		int nFeatures = firstValueIndex.length - 1;
			
		int[] featureModeValueFrequency = new int[nFeatures];
		int superModeFrequency = Integer.MIN_VALUE; 
		for(int i = 0; i < nFeatures; i++) {
			int nFeatureValue = firstValueIndex[i+1] - firstValueIndex[i];
			int modeValue = Integer.MIN_VALUE;
			for(int j = 0; j < nFeatureValue; j++) {
				if(valueFrequency[firstValueIndex[i] + j] > modeValue) {
					modeValue = valueFrequency[firstValueIndex[i] + j];
				}
			}
			featureModeValueFrequency[i] = modeValue;
			if(superModeFrequency < modeValue) {
				superModeFrequency = modeValue;
			}
		}
	
		double[] initValueOutlierness = new double[nValue];
		for(int i = 0; i < nFeatures; i++) {
			int nFeatureValue = firstValueIndex[i+1] - firstValueIndex[i];
			for(int j = 0; j < nFeatureValue; j++) {
				int frequency = valueFrequency[firstValueIndex[i] + j];
				double valueOutlierness = ((double) featureModeValueFrequency[i] - (double) frequency) / (double) featureModeValueFrequency[i]
						+ ((double) superModeFrequency - (double) featureModeValueFrequency[i]) / (double) superModeFrequency;
				valueOutlierness = 0.5 * valueOutlierness;
				initValueOutlierness[firstValueIndex[i] + j] = valueOutlierness;	
			}
		}
		return initValueOutlierness;
	}
	
	


	
	
	
	public static double[] calClustersWeight(double[][] similarityMatrix, int[] clusterInfo, 
			int nCluster, double[] valueScore) {
		double[] clusterIntraDegreeRate = new double[nCluster];
		
		for(int i = 0; i < nCluster; i++) {
			List<Integer> valueList = getValueListByCluster(clusterInfo, i);
			double intraDegreeRate = calClusterWeight(valueList, similarityMatrix, clusterInfo, valueScore);
			clusterIntraDegreeRate[i] = intraDegreeRate;
		}
		return clusterIntraDegreeRate;
	}
	public static double calClusterWeight(List<Integer> valueList, 
			double[][] similarityMatrix, int[] clusterInfo, double[] valueScore) {
		
		int nValue = clusterInfo.length;
		double[] valueDegree = new double[nValue];
		for(int i = 0; i < nValue; i++) {
			double tmpSum = 0.0;
			for(int j = 0; j < nValue; j++) {
				tmpSum += similarityMatrix[i][j];
			}
			valueDegree[i] = tmpSum;
		}
		
		double[] intraRate = new double[valueList.size()];
		for(int i = 0; i < valueList.size(); i++) {
			int nodeIndex = valueList.get(i);
			double nodeintraDegree = 0.0;
			for(int j = 0; j < valueList.size(); j++) {
				int nodeIndex2 = valueList.get(j);
				nodeintraDegree += similarityMatrix[nodeIndex][nodeIndex2];
			}
			intraRate[i] = (nodeintraDegree / valueDegree[nodeIndex]) * valueScore[nodeIndex];
		}
		
		double tmpSum = 0.0;
		for(int i = 0; i < intraRate.length; i++) {
			tmpSum += intraRate[i];
		}
		double avg = tmpSum / (double)intraRate.length;

		return avg;
	}
	
	
	public static int calcThrehold_np(int nCluster, int nValue, 
			Map<Integer, Integer> clusterSizeMap, 
			int normalClusterIndex) {
		
		int maxSize = 0;
		int remainSize = 0;		
		for(int i = 0; i < nCluster; i++) {
			int size = clusterSizeMap.get(i);
			if(size > maxSize && i != normalClusterIndex) {
				maxSize = size;
			}
			if(i != normalClusterIndex) {
				remainSize += size;
			}
		}	
		int threhold = (int) Math.round(((double)remainSize / (double)nValue)  * (double) maxSize);
		return threhold;
	}

	
	
	
	/**
	 * use cluster outlierness and cluster intrainteraction to calculate cluster weight 
	 * 
	 * @param normalClusterIndex
	 * @param clusterInterInteraction
	 * @param nCluster
	 * @return
	 */
	public static double[] getStandardClusterWeight(double[] clusterIntraDegreeRate, int nCluster, int normalClusterIndex,
			Map<Integer, Integer> clusterSizeMap,
			int threhold) {

		double[] clusterWeight = new double[nCluster];
		
		for(int i = 0; i < nCluster; i++) {
			clusterWeight[i] = clusterIntraDegreeRate[i];
			int size = clusterSizeMap.get(i);
			if(size < threhold || i == normalClusterIndex) {
				clusterWeight[i] = 0;
			}
		}
		
		return clusterWeight;
	}	

	
	
}
