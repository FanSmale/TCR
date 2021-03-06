package algorithm;

import java.util.Arrays;

import datamodel.RatingSystem2DBoolean;

/**
 * User based three-way recommender. <br>
 * Project: Three-way conversational recommendation.<br>
 * 
 * @author Fan Min<br>
 *         www.fansmale.com, github.com/fansmale/TCR.<br>
 *         Email: minfan@swpu.edu.cn, minfanphd@163.com.<br>
 * @date Created: January 31, 2019.<br>
 *       Last modified: January 31, 2020.
 * @version 1.0
 */

public abstract class UserBasedThreeWayRecommender {
	/**
	 * The dataset.
	 */
	RatingSystem2DBoolean dataset;

	/**
	 * Number of users.
	 */
	protected int numUsers;

	/**
	 * Number of items.
	 */
	protected int numItems;

	/**
	 * Number of ratings.
	 */
	protected int numRatings;

	/**
	 * Non-recommend.
	 */
	public static final int NON_RECOMMEND = 0;

	/**
	 * Promote.
	 */
	public static final int PROMOTE = 1;

	/**
	 * Recommend.
	 */
	public static final int RECOMMEND = 2;

	/**
	 * The default recommendation list length.
	 */
	public static final int DEFAULT_RECOMMENDATION_LENGTH = 10;

	/**
	 * The recommendation list length for each round.
	 */
	int recommendationLength;

	/**
	 * The default recommendation ratio.
	 */
	public static final double DEFAULT_RECOMMENDATION_RATIO = 0.3;

	/**
	 * Some items are recommended, while others are promoted.
	 */
	double recommendationRatio;

	/**
	 * The length of the recommendation list.
	 */
	int numRecommend;

	/**
	 * The length of the promotion list.
	 */
	int numPromote;

	/**
	 * The recommendations of the current user. It is shared by different
	 * recommendation algorithms. PLEASE use with caution!
	 */
	public static boolean[] currentUserRecommendations = null;

	/**
	 * The promotions of the current user. It is shared by different
	 * recommendation algorithms.
	 */
	public static boolean[] currentUserPromotions = null;

	/**
	 ************************ 
	 * The first constructor.
	 * 
	 * @param paraDataset
	 *            The given dataset.
	 ************************ 
	 */
	public UserBasedThreeWayRecommender(RatingSystem2DBoolean paraDataset) {
		dataset = paraDataset;

		initializeData();
	}// Of the first constructor

	/**
	 ************************ 
	 * Initialize some variables.
	 ************************ 
	 */
	void initializeData() {
		recommendationLength = DEFAULT_RECOMMENDATION_LENGTH;
		recommendationRatio = DEFAULT_RECOMMENDATION_RATIO;

		numUsers = dataset.getNumUsers();
		numItems = dataset.getNumItems();
		numRatings = dataset.getNumRatings();

		numRecommend = (int) (recommendationRatio * recommendationLength);
		numPromote = recommendationLength - numRecommend;
	}// Of initializeData

	/**
	 *********************************** 
	 * Setter. Recommendation length and ratio should be set simultaneously.
	 *********************************** 
	 */
	public void setRecommendationLengthRatio(int paraLength, double paraRatio) {
		recommendationLength = paraLength;
		recommendationRatio = paraRatio;

		numRecommend = (int) (recommendationRatio * recommendationLength);
		numPromote = recommendationLength - numRecommend;
	}// Of setRecommendationLength

	/**
	 *************************
	 * One round three-way recommend. Both currentUserRecommendations and
	 * currentUserPromotions should be updated here.
	 * 
	 * @param paraUser
	 *            The user.
	 * @return Recommendations/promotions of the current round. It is an integer
	 *         matrix, where the first row indicates recommended items, while
	 *         the second indicates promoted ones.
	 *************************
	 */
	public abstract int[][] threeWayRecommend(int paraUser);

	/**
	 *************************
	 * Recommend for one user.
	 * 
	 * @param paraUser
	 *            The user.
	 *************************
	 */
	public abstract void recommendForUser(int paraUser);

	/**
	 *************************
	 * Reset the recommendation and promotion lists.
	 *************************
	 */
	public void resetRecommendationPromotion() {
		if (currentUserRecommendations == null) {
			currentUserRecommendations = new boolean[numItems];
			currentUserPromotions = new boolean[numItems];
		} else {
			Arrays.fill(currentUserRecommendations, false);
			Arrays.fill(currentUserPromotions, false);
		} // Of if
	}// Of resetRecommendationPromotion

	/**
	 *********************************** 
	 * Show me.
	 *********************************** 
	 */
	public String toString() {
		String resultString = "UserBasedThreeWayRecommender on dataset:\r\n" + dataset;
		return resultString;
	}// Of toString
}// Of class UserBasedThreeWayRecommender
