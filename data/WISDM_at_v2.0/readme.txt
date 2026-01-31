Readme file for WISDM's Actitracker activity prediction data set v2.0
Updated: Oct. 22, 2013

This data has been released by the Wireless Sensor Data Mining
(WISDM) Lab. <http://www.cis.fordham.edu/wisdm/>

The data in this set were collected with our Actitracker system, 
which is available online for free at <http://actitracker.com>
and in the Google Play store.  The system is described in the 
following paper: 

Jeffrey W. Lockhart, Gary M. Weiss, Jack C. Xue, Shaun T. Gallagher, 
Andrew B. Grosner, and Tony T. Pulickal (2011). "Design Considerations
for the WISDM Smart Phone-Based Sensor Mining Architecture," Proceedings 
of the Fifth International Workshop on Knowledge Discovery from Sensor 
Data (at KDD-11), San Diego, CA. 
<http://www.cis.fordham.edu/wisdm/public_files/Lockhart-Design-SensorKDD11.pdf>

When using this dataset, we request that you cite this paper.

You may also want to cite our other relevant articles, which
can be found here:
<http://www.cis.fordham.edu/wisdm/publications.php>

Gary M. Weiss and Jeffrey W. Lockhart (2012). "The Impact of
        Personalization on Smartphone-Based Activity Recognition,"
        Proceedings of the AAAI-12 Workshop on Activity Context
        Representation: Techniques and Languages, Toronto, CA.

Jennifer R. Kwapisz, Gary M. Weiss and Samuel A. Moore (2010).
	 "Activity Recognition using Cell Phone Accelerometers,"
        Proceedings of the Fourth International Workshop on
        Knowledge Discovery from Sensor Data (at KDD-10), Washington
        DC.

When sharing or redistributing this dataset, we request that
this readme.txt file is always included.

Files:
	readme.txt
	WISDM_at_v2.0_raw_about.txt
        WISDM_at_v2.0_transformed_about.arff
        WISDM_at_v2.0_unlabeled_raw_about.txt
        WISDM_at_v2.0_unlabeled_transformed_about.arff
	WISDM_at_v2.0_demographics_about.txt 
	WISDM_at_v2.0_raw.txt
	WISDM_at_v2.0_transformed.arff
	WISDM_at_v2.0_unlabeled_raw.txt
	WISDM_at_v2.0_unlabeled_transformed.arff
	WISDM_at_v2.0_demographics.txt 


Both labeled and unlabeled data are contained in this dataset.

	Labeled data is from when the user trained Actitracker with "Training Mode" 
	The user physically specifies which activity is being performed.
	In both the raw and transformed files for labeled data, the 
	activity label is determined by the user's input.

	Unlabeled data is from when the user was running Actitracker for 
	regular use.  The user does not specify which activity is being performed.
	In the unlabeled raw data file, the activity label is "NoLabel" 
	In the unlabeled transformed file, the activity label is the activity
	that our system predicted the user to be performing.



Changelog (v2.0)
* activity label predictions added to unlabeled_transformed 


