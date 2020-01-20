package algorithm;

import java.awt.event.ItemEvent;
import java.io.IOException;

import algorithm.ThreeWayInteractiveRecommendation;
//import grale.common.SimpleTool;
//import algorithm.InteractiveRecommendation; 

/*
 * @(#)ThreeWayInteractiveRecommendation.java
 * 
 * Project: Three-way interactive recommendation.
 * Author: Fan Min, Yuan-Yuan Xu
 * www.fansmale.com
 * Email: minfan@swpu.edu.cn
 * Created: July 25, 2019.
 * Last modified: July 25, 2019.
 */

public class popularityBasedRecommend {

	/**
	 * The number of items.
	 *
	 */
	int numItems;

	/**
	 * The number of ratings.
	 *
	 */
	int numRatings;

	/**
	 * The threshold of popularity.
	 */
	int popThreshold;

	/**
	 * The length of recommendation list.
	 */
	int pageLength;

	/**
	 * Record the items which are recommended or promoted
	 */
	int[] greyList;

	/**
	 * 
	 * @param paraTir
	 * @param paraUserId
	 * @param parapageLength
	 * @param paraBoughtList
	 * @param paraThreshold
	 */

	public int []popularityBasedRecommend(ThreeWayInteractiveRecommendation paraTir, int paraUserId, int paraPageLength,
//			public popularityBasedRecommend(ThreeWayInteractiveRecommendation paraTir, int paraUserId, int paraPageLength,
			//			int[] paraBoughtList, int paraPopThreshold, int[] paraGreyList, double paraProOfRec) {
			boolean[] paraBoughtList, int paraPopThreshold, int[] paraGreyList, double paraProOfRec) {
		pageLength = paraPageLength;
		popThreshold = paraPopThreshold;
		numItems = paraTir.numItems;
		greyList = paraGreyList;

		int[] tempRecList = new int[numItems];
		int[] tempItemPops = new int[numItems];

		int tempRecLen = (int) (pageLength * paraProOfRec);
		int tempProLen = pageLength - tempRecLen;

		int tempRecCount = 0;
		int tempProCount = 0;

		// Step 1: randomly sort the indices of items.
		randomItemIndices();

		// Step 2: get the current popularity
		tempItemPops = getItemPops(paraUserId, paraPageLength, paraBoughtList, paraTir);

		// Step 3:
		for (int i = 0; i < numItems; i++) {
			if (paraGreyList[i] == 1) {
				continue;
			} // Of if

			if (tempItemPops[i] >= popThreshold && tempRecCount < tempRecLen) {
				tempRecList[i] = 2;
				tempRecCount++;
			} else if (tempItemPops[i] < popThreshold && tempProCount < tempProLen) {
				tempRecList[i] = 1;
				tempProCount++;
			} // Of if

			if (tempRecCount + tempProCount == pageLength) {
				break;
			} // Of if
		} // Of for i

		return tempRecList;
	}// Of popularityBasedRecommend

	/**
	 *********************************** 
	 * Randomly sort the indices of items.
	 *********************************** 
	 */
	public void randomItemIndices() {

	}

	/**
	 *********************************** 
	 * Recompute the popularities of items after leaving one user out.
	 *********************************** 
	 */
	public int[] getItemPops(int paraUserId, int parapageLength, boolean[]
	 paraBoughtList, ThreeWayInteractiveRecommendation paraTir) {
	
	 int[] tempItemPops = new int[numItems];
	 for (int i = 0; i < numRatings; i++) {
		 if (paraTir.userVector[i] != paraUserId && paraTir.ratingVector[i] > 1e-6) {
			 	tempItemPops[paraTir.itemVector[i]]++;
		 	} // Of if
	
		 if (paraBoughtList[paraTir.itemVector[i]] == true) {
		 		tempItemPops[paraTir.itemVector[i]]++;
	 		} // Of if
	 	} // Of for i
	 return tempItemPops;
	 }// Of getItemPops


}// Of Class AveRateRec
