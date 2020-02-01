package algorithm;

import common.SimpleTools;

import datamodel.*;

import java.io.*;
import java.util.Arrays;

/**
 * Popularity-based recommendation. <br>
 * Project: Three-way conversational recommendation.<br>
 * 
 * @author Fan Min<br>
 *         www.fansmale.com, github.com/fansmale/TCR.<br>
 *         Email: minfan@swpu.edu.cn, minfanphd@163.com.<br>
 * @date Created: January 31, 2019.<br>
 *       Last modified: January 31, 2020.
 * @version 1.0
 */

public class PopularityBasedRecommendation extends UserBasedThreeWayRecommender {

	/**
	 * The maturity value array. For each recommendation to existing rating, the
	 * maturity increases by 5. For each recommendation to non-existing rating,
	 * the maturity increases by 2.5. For each promotion to existing rating, the
	 * maturity increases by 4. For each promotion to non-existing rating, the
	 * maturity increases by 2.
	 */
	double[] maturityValueArray = { 5, 2.5, 4, 2 };

	/**
	 * The default maturity threshold.
	 */
	public static final double DEFAULT_MATURITY_THRESHOLD = 1000;

	/**
	 * The maturity threshold.
	 */
	double maturityThreshold;

	/**
	 * The default maturity value for each item.
	 */
	public static final double DEFAULT_MATURITY_FOR_EACH_ITEM = 3;

	/**
	 * The recommendation list for the current user. boolean[]
	 * currentUserRecommendations;
	 */

	/**
	 * The promotion list for the current user. boolean[] currentUserPromotions;
	 */

	/**
	 * The maximum of item pop.
	 */
	int maxItemPopularity;

	/**
	 * Item popularity no less than the second value may be recommended, no less
	 * than the first value may be promoted.
	 */
	double[] popularityThresholds = { 0.5, 0.9 };

	/**
	 * The popular items (for recommend). It is user related since the current
	 * user cannot be considered.
	 */
	private int[] popularItems;

	/**
	 * The semi-popular items (for promote). It is user related since the
	 * current user cannot be considered.
	 */
	private int[] semiPopularItems;

	/**
	 * The statistics information (NN, NP, BN, ...) for current user. int[][]
	 * recommendationStatistics = new int[3][2];
	 */

	/**
	 *********************************** 
	 * The constructor.
	 * 
	 * @param paraFilename
	 *            The rating filename.
	 * @param paraNumUsers
	 *            The number of users.
	 * @param paraNumItems
	 *            The number of items.
	 * @param paraNumRatings
	 *            The number of ratings.
	 * @param paraRatingLowerBound
	 *            The lower bound of ratings.
	 * @param paraRatingUpperBound
	 *            The upper bound of ratings.
	 * @param paraCompress
	 *            Is the data in compress format?
	 *********************************** 
	 */
	public PopularityBasedRecommendation(String paraFilename, int paraNumUsers, int paraNumItems,
			int paraNumRatings, double paraRatingLowerBound, double paraRatingUpperBound,
			boolean paraCompress) {
		// Step 1. Read data.
		super(paraFilename, paraNumUsers, paraNumItems, paraNumRatings, paraRatingLowerBound,
				paraRatingUpperBound, paraCompress);

		// Step 3. Initialize.
		initialize();
	}// Of the constructor

	/**
	 ************************ 
	 * The second constructor.
	 * 
	 * @param paraDataset
	 *            The given dataset.
	 ************************ 
	 */
	public PopularityBasedRecommendation(RatingSystem2DBoolean paraDataset) {
		super(paraDataset);

		initialize();
	}// Of the second constructor

	/**
	 *********************************** 
	 * Initialize.
	 *********************************** 
	 */
	public void initialize() {
		maturityThreshold = DEFAULT_MATURITY_THRESHOLD;

		maxItemPopularity = computeMaxItemPopularity();

		computePopAndSemipopItems();
	}// Of initialize

	/**
	 *********************************** 
	 * Setter.
	 * 
	 * @param paraArray
	 *            The maturity value array with four values.
	 *********************************** 
	 */
	public void setMaturityValueArray(double[] paraArray) {
		maturityValueArray = paraArray;
	}// Of setMaturityValueArray

	/**
	 *********************************** 
	 * Getter.
	 *********************************** 
	 * public int[][] getRecommendationStatistics() { return
	 * recommendationStatistics; }// Of getRecommendationStatistics
	 */

	/**
	 *********************************** 
	 * Setter.
	 *********************************** 
	 */
	public void setMaturityThreshold(double paraValue) {
		maturityThreshold = paraValue;
	}// Of setMaturityThreshold

	/**
	 *********************************** 
	 * Setter.
	 * 
	 * @param paraThresholds
	 *            Should have exactly two elements.
	 *********************************** 
	 */
	public void setPopularityThresholds(double[] paraThresholds) {
		popularityThresholds = paraThresholds;
	}// Of setPopularityThresholds

	/**
	 *********************************** 
	 * Compute both popular items, which may be recommended, and semi-popular
	 * items, which may be promoted.
	 *********************************** 
	 */
	public void computePopAndSemipopItems() {
		computePopAndSemipopItems(popularityThresholds[1], popularityThresholds[0]);
	}// Of computePopAndSemipopItems

	/**
	 *********************************** 
	 * Compute both popular items, which may be recommended, and semi-popular
	 * items, which may be promoted.
	 *********************************** 
	 */
	public void computePopAndSemipopItems(double paraPopThreshold, double paraSemiPopThreshold) {
		// Step 1. Compute popular items.
		int tempNumPopItems = 0;
		// Step 1.1 Compute the length
		for (int i = 0; i < numItems; i++) {
			if (dataset.getItemPopularity(i) > maxItemPopularity * paraPopThreshold) {
				tempNumPopItems++;
			} // Of if
		} // Of for i

		// Step 1.2 Compute the array
		popularItems = new int[tempNumPopItems];
		tempNumPopItems = 0;
		for (int i = 0; i < numItems; i++) {
			if (dataset.getItemPopularity(i) > maxItemPopularity * paraPopThreshold) {
				popularItems[tempNumPopItems] = i;
				tempNumPopItems++;
			} // Of if
		} // Of for i

		// Step 2. Compute semi-popular items.
		int tempNumSemiPopItems = 0;
		// Step 2.1 Compute the length
		for (int i = 0; i < numItems; i++) {
			if ((dataset.getItemPopularity(i) > maxItemPopularity * paraSemiPopThreshold)
					&& (dataset.getItemPopularity(i) <= maxItemPopularity * paraPopThreshold)) {
				tempNumSemiPopItems++;
			} // Of if
		} // Of for i

		// Step 1.2 Compute the array
		semiPopularItems = new int[tempNumSemiPopItems];
		tempNumSemiPopItems = 0;
		for (int i = 0; i < numItems; i++) {
			if ((dataset.getItemPopularity(i) > maxItemPopularity * paraSemiPopThreshold)
					&& (dataset.getItemPopularity(i) <= maxItemPopularity * paraPopThreshold)) {
				semiPopularItems[tempNumSemiPopItems] = i;
				tempNumSemiPopItems++;
			} // Of if
		} // Of for i

		SimpleTools.variableTrackingOutput("Pop items: " + Arrays.toString(popularItems));
		SimpleTools
				.variableTrackingOutput("Semi-popular items: " + Arrays.toString(semiPopularItems));
	}// Of computePopAndSemipopItems

	/**
	 *********************************** 
	 * Compute the maximum of item pop.
	 *********************************** 
	 */
	public int computeMaxItemPopularity() {
		int resultMaxPop = dataset.getItemPopularity(0);
		for (int i = 1; i < numItems; i++) {
			if (resultMaxPop < dataset.getItemPopularity(i)) {
				resultMaxPop = dataset.getItemPopularity(i);
			} // Of if
		} // Of for i
		return resultMaxPop;
	}// Of computeMaxItemPopularity

	/**
	 *********************************** 
	 * Recommend for one user.
	 * 
	 * @param paraUser
	 *            The user index.
	 * @return Recommendations and promotions.
	 *********************************** 
	 */
	public boolean[][] recommendForUser(int paraUser) {
		SimpleTools.processTrackingOutput("\r\nUser " + paraUser);

		// Step 1. Initialize the total/average score of each item.
		int[] tempItemPopArray = new int[numItems];
		for (int i = 0; i < numItems; i++) {
			tempItemPopArray[i] = dataset.getItemPopularity(i);
		} // Of for i

		// Remove those for the current user since the data is unknown.
		for (int i = 0; i < dataset.getUserNumRatings(paraUser); i++) {
			tempItemPopArray[dataset.getTriple(paraUser, i).item]--;
		} // Of for i

		// Step 2. Compute popular items.
		double tempMaturity = 0;
		int[] tempPopItems = new int[numItems];
		int tempNumPopItems = 0;
		// System.out.print("Popular items: ");
		for (int i = 0; i < numItems; i++) {
			if (tempItemPopArray[i] >= popularityThresholds[1] * maxItemPopularity) {
				tempPopItems[tempNumPopItems] = i;
				// System.out.print(", " + i);
				tempNumPopItems++;
			} // Of for i
		} // Of for i

		// Step 3. Compute semi-popular items.
		int[] tempSemiPopItems = new int[numItems];
		int tempNumSemiPopItems = 0;
		// System.out.print("\r\nSemi-popular items: ");
		for (int i = 0; i < numItems; i++) {
			if ((tempItemPopArray[i] < popularityThresholds[1] * maxItemPopularity)
					&& (tempItemPopArray[i] >= popularityThresholds[0] * maxItemPopularity)) {
				tempSemiPopItems[tempNumSemiPopItems] = i;
				// System.out.print("; " + i);
				tempNumSemiPopItems++;
			} // Of for i
		} // Of for i

		// Step 4. Initialize
		boolean[] tempCurrentUserRecommendations = new boolean[numItems];
		boolean[] tempCurrentUserPromotions = new boolean[numItems];

		int[][] tempRecommendPromote = null;

		while ((tempMaturity < maturityThreshold)) {
			tempRecommendPromote = threeWayRecommend(paraUser, tempCurrentUserRecommendations,
					tempCurrentUserPromotions);

			if (tempRecommendPromote == null) {
				break;
			} else {
				SimpleTools.variableTrackingOutput("recommendation/promotion for user " + paraUser
						+ ": " + Arrays.deepToString(tempRecommendPromote));
			} // Of if

			// Step 3 Update the maturity
			tempMaturity += computeUserMaturity(paraUser, tempCurrentUserRecommendations,
					tempCurrentUserPromotions);
		} // Of while

		if (tempMaturity >= maturityThreshold) {
			SimpleTools.processTrackingOutput("Matured.");
		} else {
			SimpleTools.processTrackingOutput("The maturity " + tempMaturity
					+ " is smaller than the threshold " + maturityThreshold);
		} // Of if

		boolean[][] resultRecommendationPromotions = new boolean[2][];
		resultRecommendationPromotions[0] = tempCurrentUserRecommendations;
		resultRecommendationPromotions[1] = tempCurrentUserPromotions;
		return resultRecommendationPromotions;
	}// Of recommendForUser

	/**
	 *********************************** 
	 * Pop-based recommendation. With member variables
	 * currentUserRecommendations and currentUserPromotions, there is no need to
	 * return.
	 * 
	 * @param paraUser
	 *            The given user.
	 *********************************** 
	 */
	public int[][] threeWayRecommend(int paraUser, boolean[] paraRecommendations,
			boolean[] paraPromotions) {

		// Step 1. Find popular however not recommended items.
		int[] tempPopularUnrecommendedItems = new int[popularItems.length];
		int tempCounter = 0;
		for (int i = 0; i < popularItems.length; i++) {
			if (paraRecommendations[popularItems[i]]) {
				continue;
			} // Of if

			tempPopularUnrecommendedItems[tempCounter] = popularItems[i];
			tempCounter++;
		} // Of for i

		if (numRecommend > tempCounter) {
			SimpleTools.processTrackingOutput("No enough to recommend.");
			return null;
		} // Of if

		// Step 2. Randomly select some to recommend.
		int[] tempRecommendations = SimpleTools.randomSelectFromArray(tempPopularUnrecommendedItems,
				tempCounter, numRecommend);
		for (int i = 0; i < tempRecommendations.length; i++) {
			paraRecommendations[tempRecommendations[i]] = true;
		} // Of for i

		// Step 3. Find semi-popular however unpromoted items.
		int[] tempSemiPopularUnpromotedItems = new int[semiPopularItems.length];
		tempCounter = 0;
		for (int i = 0; i < semiPopularItems.length; i++) {
			if (paraPromotions[semiPopularItems[i]]) {
				continue;
			} // Of if

			tempSemiPopularUnpromotedItems[tempCounter] = semiPopularItems[i];
			tempCounter++;
		} // Of for i

		if (numPromote > tempCounter) {
			SimpleTools.processTrackingOutput("No enough to promote.");
			return null;
		} // Of if

		// Step 4. Randomly select some to promote
		int[] tempPromotions = SimpleTools.randomSelectFromArray(tempSemiPopularUnpromotedItems,
				tempCounter, numPromote);
		for (int i = 0; i < tempPromotions.length; i++) {
			paraPromotions[tempPromotions[i]] = true;
		} // Of for i

		// Step 5. Construct the lists.
		int[][] resultArrays = new int[2][];
		resultArrays[0] = tempRecommendations;
		resultArrays[1] = tempPromotions;

		return resultArrays;
	}// Of threeWayRecommend

	/**
	 *********************************** 
	 * Eliminate elements that are already processed.
	 * 
	 * @param paraArray
	 *            The given array.
	 * @param paraProcessed
	 *            Indicate which elements have been processed.
	 * @return The new array.
	 *********************************** 
	 */
	public static int[] eliminateProcessed(int[] paraArray, boolean[] paraProcessed) {
		int[] tempArray = new int[paraArray.length];
		int tempLength = 0;

		// Step 1. Copy unprocessed elements.
		for (int i = 0; i < paraArray.length; i++) {
			if (!paraProcessed[paraArray[i]]) {
				tempArray[tempLength] = paraArray[i];
				tempLength++;
			} // Of if
		} // Of if

		// Step 2. Compress.
		int[] resultArray = new int[tempLength];
		for (int i = 0; i < tempLength; i++) {
			resultArray[i] = tempArray[i];
		} // Of for i
		return resultArray;
	}// Of eliminateProcessed

	/**
	 *********************************** 
	 * Compute the user maturity increment. For each recommendation to existing
	 * rating, the maturity increases by 5. For each recommendation to
	 * non-existing rating, the maturity increases by 2.5. For each promotion to
	 * existing rating, the maturity increases by 4. For each promotion to
	 * non-existing rating, the maturity increases by 2.
	 * 
	 * @param paraUser
	 *            The given user.
	 * @param paraRecommendations
	 *            The recommendation array.
	 * @param paraPromotions
	 *            The promotion array.
	 *********************************** 
	 */
	public double computeUserMaturity(int paraUser, boolean[] paraRecommendations,
			boolean[] paraPromotions) {
		double resultValue = 0;
		// Step 1. Which items are actually rated.
		boolean[] tempUserBehaviors = new boolean[numItems];
		for (int i = 0; i < dataset.getUserNumRatings(paraUser); i++) {
			tempUserBehaviors[dataset.getTriple(paraUser, i).item] = true;
		} // Of for i

		// 0 for no recommendation/promotion
		for (int i = 0; i < numItems; i++) {
			if (paraRecommendations[i]) {
				if (tempUserBehaviors[i]) {
					resultValue += maturityValueArray[0];
				} else {
					resultValue += maturityValueArray[1];
				} // Of if
			} else if (paraPromotions[i]) {
				if (tempUserBehaviors[i]) {
					resultValue += maturityValueArray[2];
				} else {
					resultValue += maturityValueArray[3];
				} // Of if
			} // Of for i
		} // Of for i

		return resultValue;
	}// Of computeUserMaturity

	/**
	 *********************************** 
	 * The main entrance.
	 * 
	 * @throws IOException
	 * @throws NumberFormatException
	 *********************************** 
	 */
	public static void testPopularityBasedRecommendation() {
		SimpleTools.variableTracking = true;
		// TIR2 tir = new TIR2("data/movielens100k.data", 943, 1682, 100000,
		// -10, 10);
		PopularityBasedRecommendation tempPbr = new PopularityBasedRecommendation(
				"data/jester-data-1/jester-data-1.txt", 24983, 101, 1810455, -10, 10, false);
		System.out.println(tempPbr);

		tempPbr.setPopularityThresholds(new double[] { 0.6, 0.7 });
		tempPbr.computePopAndSemipopItems();

		tempPbr.recommendForUser(2);
	}// Of testPopularityBasedRecommendation

	/**
	 *********************************** 
	 * The main entrance.
	 * 
	 * @throws IOException
	 * @throws NumberFormatException
	 *********************************** 
	 */
	public static void main(String args[]) {
		SimpleTools.processTracking = true;
		testPopularityBasedRecommendation();
	}// Of main

}// Of class TIR