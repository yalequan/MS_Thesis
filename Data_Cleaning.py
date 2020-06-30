import pandas as pd
import numpy as np

"""
We begin by importing the initial dataset along with an update which includes 
the students High School GPA. The high school GPA datset is subset so it only 
includes ID and GPA. This is then merged in ID# with the main dataset
"""
thesis_df = pd.read_csv(r"C:\Users\yaleq\OneDrive - csulb\Yale_Quan_Thesis\Python_Code\IRAReport1687_2.csv", low_memory=False)

thesis_df = thesis_df.sort_values(by = ['emplid'])

hs_gpa = pd.read_csv(r"C:\Users\yaleq\OneDrive - csulb\Yale_Quan_Thesis\Python_Code\gpa_data.csv", low_memory=False)

hs_gpa = hs_gpa[['emplid','hs_gpa']]

thesis_df = pd.merge(thesis_df, hs_gpa, on = 'emplid') 

del hs_gpa

thesis_df = thesis_df.sort_values(by=['emplid', 'class_term'])

"""
Students who are classified as Young Scholars (YSCHOT00OU), Early Entrant
(ERLYOT00OU), Non-Degree Seeking Students, and International Students who did 
not take the ACT or SAT were removed from the dataset. These students are 
considered non-traditional students and fall outside the scope of our study. 
They will be removed at the end of the analysis
"""

# a) Identify Students who are young scholars plan code YSCHOT00OU

young_scholars = thesis_df[thesis_df["student_acad_plan_code"]\
                           .str.match('YSCHOT00OU', case = False)]

young_scholars = young_scholars.emplid.unique().tolist()

# b) Identify Early Entrant students plan code ERLYOT00OU

early_entrant = thesis_df[thesis_df["student_acad_plan_code"]\
                          .str.match('ERLYOT00OU', case = False)]

early_entrant = early_entrant.emplid.unique().tolist()

# c) Identify Non-Degree seeking students

non_degree = thesis_df[thesis_df["student_acad_plan"]\
                       .str.contains('Non Degree', case = False)]

non_degree = non_degree.emplid.unique().tolist()

# d) Identify International students and student who did not take SAT or ACT

international = thesis_df[(thesis_df["sat_comp"]==0) &\
                          (thesis_df["act_composite_score"]==0)]

"""
The data are currently organized as one row per course taken in a given 
semester. To make the data easier to analyze and understand we create a new 
ataframe (student_data) where each row is a unique student. We begin by 
creating a list of all unique student ID# and then using those to 
subset the original dataset.
"""

# List of Unique Student ID
unique_id = thesis_df.emplid.unique().tolist()

# Create a dataframe of Unique Student ID that columns can be appended to

student_data = pd.DataFrame(unique_id, columns = ['emplid'])

del unique_id


# Append Winter/Summer Flag

student_data['winter_summer_flag'] = thesis_df.groupby\
    ('emplid')['winter_summer_flag'].max().to_list()

# Append High School GPA

student_data['High_School_GPA'] = thesis_df.groupby\
    ('emplid')['hs_gpa'].first().to_list()

# Parent Eduation

student_data['parent_education_1'] = thesis_df.groupby\
    ('emplid')['parent_education_1'].first().to_list()

student_data['parent_education_2'] = thesis_df.groupby\
    ('emplid')['parent_education_2'].first().to_list()

# Pell Eligibility

student_data['pell_eligibility'] = thesis_df.groupby\
    ('emplid')['pell_eligibility'].first().to_list()

# Local Student status for Admission Requirements

student_data['lsa_local_grouping'] = thesis_df.groupby\
    ('emplid')['lsa_local_grouping'].first().to_list()

# Minority Status based on admission declaration

student_data['minority'] = thesis_df.groupby\
    ('emplid')['minority'].first().to_list()

# Cohort Year

student_data['cohort'] = thesis_df.groupby\
    ('emplid')['year_enrolled'].min().to_list()

# Exit Year

student_data['exit_year'] = thesis_df.groupby\
    ('emplid')['year_enrolled'].max().to_list()

# How many years the student completed until exiting the university

student_data['years'] = student_data['exit_year'] - student_data['cohort']

####################################################

major_df = thesis_df[['emplid', 'class_term', 'student_acad_plan_code']]

major_df = major_df.sort_values(by=['emplid', 'class_term'])

# Starting Major Code

start_df = major_df.groupby\
    ('emplid')['class_term','student_acad_plan_code'].first()

start_df.reset_index(level=0, inplace=True)

start_df = start_df.drop(columns=['class_term'])

start_df.columns = ['emplid', 'start_major']

# End major

end_df = major_df.groupby\
    ('emplid')['class_term','student_acad_plan_code'].last()
    
end_df.reset_index(level=0, inplace=True)

end_df = end_df.drop(columns=['class_term'])

end_df.columns = ['emplid', 'end_major']
    

student_data = pd.merge(student_data, start_df, on = 'emplid') 
student_data = pd.merge(student_data, end_df, on = 'emplid') 

###########################################################################

college_df = thesis_df[['emplid', 'class_term', 'student_college_code']]

college_df = college_df.sort_values(by=['emplid', 'class_term'])


# Starting College 

start_df = college_df.groupby\
    ('emplid')['class_term','student_college_code'].first()

start_df.reset_index(level=0, inplace=True)

start_df = start_df.drop(columns=['class_term'])

start_df.columns = ['emplid', 'start_college']

# End College

end_df = college_df.groupby\
    ('emplid')['class_term','student_college_code'].last()
    
end_df.reset_index(level=0, inplace=True)

end_df = end_df.drop(columns=['class_term'])

end_df.columns = ['emplid', 'end_college']
    

student_data = pd.merge(student_data, start_df, on = 'emplid') 
student_data = pd.merge(student_data, end_df, on = 'emplid') 


"""
To calculate the number of semesters enrolled the data are first organized so 
each row represents 1 semester. Then the data are grouped by Student ID# and 
the number of semester rows are counted.
"""

#(1) Subset the full data so each row represents a semester

semester_df = thesis_df.drop_duplicates(subset=['emplid', 'class_term'])

semester_df = semester_df.sort_values(by=['emplid', 'class_term'])

#(2) Count each semester

student_data['semesters'] = semester_df.groupby('emplid')['class_term'].\
    count().to_list()

"""
Students who are enrolled in majors under COE, CNSM, and ES&P are considered 
STEM Majors. To filter these students out a filter was written that captures 
students who are majoring in COE or CNSM or COE. This filter is then applied 
to the starting college and ending college column.
"""

# STEM entrant or not

student_data['STEM_Entrant'] = np.where((student_data['start_college']=='COE')|
            (student_data['start_college']=='CNSM')|
            (student_data['start_college']=='ES&P'),1,0)

# STEM graduate or not

student_data['STEM_Grad'] = np.where((student_data['end_college']=='COE')|
            (student_data['start_college']=='CNSM')|
            (student_data['start_college']=='ES&P'),1,0)

"""
Columns were created which contain the major for a student in a particuler 
semester. Students with NAN are students who are not enrolled that semester.
"""

# Create a dataframe of all majors a student had

# Create a dataframe of all majors a student had

major_list = semester_df.groupby('emplid')\
                                 ['student_acad_plan_code'].apply(list)

major_list_df = pd.DataFrame(major_list)

major_list_df = major_list_df['student_acad_plan_code'].apply(pd.Series)

#This df we can merge with the student data to give us 
#a major per semester after renaming for easier understanding

columns = list(major_list_df)

A = 'Semester'

merge_df = pd.DataFrame()
for i in columns:
    merge_df[A + "_" + str(i+1) +"_Major"] = major_list_df.iloc[:,i]

# Reset the index to have emplid to merge on
merge_df = merge_df.reset_index().sort_values(by = ['emplid'])

student_data = pd.merge(student_data, merge_df, on = 'emplid')

"""
Overall GPA
"""
Overall_GPA = thesis_df.groupby('emplid')\
    ['enrl_units_taken','enrl_grade_points'].sum()

Overall_GPA['Overall_GPA'] =\
    round(Overall_GPA['enrl_grade_points']/Overall_GPA['enrl_units_taken'],2)

Overall_GPA = Overall_GPA.drop\
    (['enrl_grade_points','enrl_units_taken'], axis = 1)

Overall_GPA = Overall_GPA.reset_index().sort_values(by = ['emplid'])

student_data = pd.merge(student_data, Overall_GPA, on = 'emplid')

"""
Semester GPA

Student's were grouped by ID and class term and the units taken and 
grade points were summed for each semester. This was exported to Excel 
were a pivot table was used to create the final seemster GPA data
"""
Semester_GPA = thesis_df.groupby \
(['emplid','class_term'])['enrl_units_taken','enrl_grade_points'].sum()

Semester_GPA['Semester_GPA'] =\
    round(Semester_GPA['enrl_grade_points']/Semester_GPA['enrl_units_taken'],2)

Semester_GPA =\
    Semester_GPA.drop(['enrl_grade_points', 'enrl_units_taken'], axis = 1)
    
#Semester_GPA.to_csv(r'C:\Users\yaleq\OneDrive\Desktop\Semester_GPA.csv', index = True, header=True)

semester_GPA_Cleaned = pd.read_csv(r"C:\Users\yaleq\OneDrive - csulb\Yale_Quan_Thesis\Python_Code\semester_gpa_cleaned.csv")

student_data = pd.merge(student_data, semester_GPA_Cleaned, on = 'emplid')
"""
Standardized test scores.

The data was subset so each row is a student and then subset again to only 
include standardized test scores.  Students who did not take standardized 
exams were dropped.
"""
# Subset data so each row is a student

student_row = thesis_df.drop_duplicates(subset=['emplid'])

# Subset so the data is only test scores and the unique ID

test_scores = student_row[['emplid',
                            'SAT_NEW_Math',
                            'SAT_NEW_Reading',
                            'SAT_NEW_Conversion_Composite',
                            'ACT_Reading_Conversion',
                            'ACT_Math_Conversion',
                            'ACT_Composit_Conversion',
                            'ACT_Composit_Conversion',
                            'MAX_ACT_SAT_MATH',
                            'MAX_ACT_SAT_Reading']]

student_data = pd.merge(student_data, test_scores, on = 'emplid')


"""
Eligibility index scores were calculated using the formulas provided 
on https://www.csulb.edu/admissions/freshmen-eligibility-index
"""

student_data['Eligibility_Index'] = np.where((student_data
            ['STEM_Entrant'] == 0),800 * student_data['High_School_GPA'] +
            student_data['MAX_ACT_SAT_Reading'] + 
            student_data['MAX_ACT_SAT_MATH'],600 *\
                student_data['High_School_GPA'] +
            student_data['MAX_ACT_SAT_Reading'] + 
            2 * student_data['MAX_ACT_SAT_MATH'])

"""
Student Gender
"""

gender_df = student_row[['emplid','sex_code']]

gender_df.columns = ['emplid', 'gender']

gender_df = gender_df.sort_values(by = ['emplid'])

student_data = pd.merge(student_data, gender_df, on = 'emplid')

# Binary coding for gender

def gender_func(x):
    if x == 'M':
        return 0
    return 1

student_data['gender_code'] = student_data['gender'].apply(gender_func)

"""
GE units taken

The total amount of GE and Non-GE Units were calculated for each student.  
Another datframe was created which has the semester level information. 
This was then exported to Excel where pivot tables were used to clean
"""

# Total GE units

Total_GE_Units = thesis_df.groupby \
('emplid')['GE_Credits','Non_GE_Credit'].sum()

Total_GE_Units = Total_GE_Units.reset_index().sort_values(by = ['emplid'])


student_data = pd.merge(student_data, Total_GE_Units, on = 'emplid')

# Semester GE units

Semester_GE = thesis_df.groupby(['emplid','class_term'])\
    ['GE_Credits','Non_GE_Credit'].sum()

# A pivot table in excel was created to merge the semester GE credits

per_semester_GE = pd.read_csv(r"C:\Users\yaleq\OneDrive - csulb\Yale_Quan_Thesis\Python_Code\GE_Credits.csv")

per_semester_Non_GE = pd.read_csv(r"C:\Users\yaleq\OneDrive - csulb\Yale_Quan_Thesis\Python_Code\Non_GE_Credits.csv")

student_data = pd.merge(student_data, per_semester_GE, on = 'emplid')
student_data = pd.merge(student_data, per_semester_Non_GE, on = 'emplid')

"""
Graduated Flag
"""

graduated = student_row[['emplid','graduated_flag']]

graduated.graduated_flag.replace(('Y', 'N'), (1, 0), inplace=True)

student_data = pd.merge(student_data, graduated, on = 'emplid')

"""
Number of Major Changes
"""
# The first things is to create a df which contains the amount of semesters 
# a student was undeclared before declaring a major

undeclared_time = major_list_df.copy()

count_df = pd.DataFrame()
for i in columns:
    count_df[A + "_" + str(i+1)] = undeclared_time.iloc[:,i].apply(
        lambda x: -1 if x == 'NDUGOT00U1' else i)

    
undeclared_time_count = count_df.replace(-1, np.nan).bfill(1)\
.iloc[:, 0].to_frame().reset_index()

# Rename the column

undeclared_time_count = undeclared_time_count.\
rename(columns = {'Semester_1':'Semesters_Undeclared'})

# Merge with student data

student_data = pd.merge(student_data, undeclared_time_count, on = 'emplid')

# Export to Excel to count levels (number of major changes). 
# Note that students who did not change will have a level count of 1 major
# We will subtract 1 from all cells that have only 1 to ensure that
# they are logged as zero changes

# Then we will figure out if they had semesters of undeclared before declaring
# if they do we subtract 1 from the levels

# major_list_df_transposed = major_list_df.T
# print(len(major_list_df_transposed.columns))

# df1 = major_list_df_transposed.iloc[:, :4000]
# df2 = major_list_df_transposed.iloc[:, 4000:8000]
# df3 = major_list_df_transposed.iloc[:, 8000:12000]
# df4 = major_list_df_transposed.iloc[:, 12000:]

# df1.to_csv(r'C:\Users\yaleq\OneDrive\Desktop\major_list_df_1.csv', 
#            index = True, header=True)
# df2.to_csv(r'C:\Users\yaleq\OneDrive\Desktop\major_list_df_2.csv', 
#            index = True, header=True)
# df3.to_csv(r'C:\Users\yaleq\OneDrive\Desktop\major_list_df_3.csv', 
#            index = True, header=True)
# df4.to_csv(r'C:\Users\yaleq\OneDrive\Desktop\major_list_df_4.csv', 
#            index = True, header=True)

major_levels = pd.read_csv(r"C:\Users\yaleq\OneDrive - csulb\Yale_Quan_Thesis\Python_Code\Major_Levels.csv")

adjust = -1

major_levels['Adjusted'] = major_levels['Number_of_Levels'].\
where(major_levels['Number_of_Levels'] != 1,
      major_levels['Number_of_Levels']+adjust)

# Need a flag for students who started as undeclared to fix the count

Start_Undeclared_Flag = undeclared_time_count.copy()

Start_Undeclared_Flag[Start_Undeclared_Flag != 0] = 1

major_levels['Undeclared_Start'] =\
Start_Undeclared_Flag['Semesters_Undeclared']

major_levels['Number_of_Major_Changes'] =\
major_levels["Adjusted"] - major_levels["Undeclared_Start"]

# This results in a value of -1 if a students started as undeclared and 
# never declared a major.  We will fix this by changing all of these to zero

major_levels['Number_of_Major_Changes']=\
major_levels['Number_of_Major_Changes'].mask(major_levels\
            ['Number_of_Major_Changes'].lt(0),0)

# Add this to the student data

    
major_levels = major_levels[['emplid', 'Number_of_Major_Changes']]\
.sort_values(by = ['emplid'])  
    
student_data = pd.merge(student_data, major_levels, on = 'emplid')

"""
Undeclared Start
"""
def undeclared_func(x):
    if x == 0:
        return 0
    return 1

student_data['undeclared_start'] = student_data['Semesters_Undeclared']\
    .apply(undeclared_func)

"""
Major Change Levels
"""
def major_change_level_func(x):
    if x == 0:
        return 'A'
    elif x == 1:
        return 'B'
    elif x == 2:
        return 'C'
    return 'D'

student_data['major_change_levels'] = student_data['Number_of_Major_Changes']\
    .apply(major_change_level_func)

"""
Major Change Flag
"""
def major_change_flag_func(x):
    if x == 0:
        return 0
    return 1

student_data['major_change_flag'] = student_data['Number_of_Major_Changes']\
    .apply(major_change_flag_func)

"""
Major Change indicator variable per semester
"""

Semester_Major_Change = pd.read_csv(r"C:\Users\yaleq\OneDrive - csulb\Yale_Quan_Thesis\Python_Code\Semester_Major_Change_Indicator.csv")

student_data = pd.merge(student_data, Semester_Major_Change, on = 'emplid')

"""
DFW Grades
"""

DFW = thesis_df.groupby(['emplid', 'class_term',"enrl_official_grade"])\
    ['enrl_official_grade'].count()
    
DFW_Counts= pd.read_csv(r"C:\Users\yaleq\OneDrive - csulb\Yale_Quan_Thesis\Python_Code\DFW_Counts.csv")

# Merge with student data

student_data = pd.merge(student_data, DFW_Counts, on = 'emplid')   
    
"""
Rolling GPA Change
"""

semester_gpa = student_data.iloc[:, 0:69]
semester_gpa.drop(semester_gpa.iloc[:, 1:44], inplace = True, axis = 1)

# Check to make sure the right columns were selected

list(semester_gpa.columns)
    
# Calculate rolling diff
rolling_difference = semester_gpa.iloc[:,1:26].diff(axis = 1)
rolling_difference["emplid"] = semester_gpa["emplid"]  
    
# Semester 1 will be blank because there can't be a change between 
#semester 0 and semester 1

# rolling_difference.to_csv(r'C:\Users\yaleq\OneDrive\Desktop\rolling_difference.csv', 
#             index = True, header=True)

rolling_difference_cleaned= pd.read_csv(r"C:\Users\yaleq\OneDrive - csulb\Yale_Quan_Thesis\Python_Code\rolling_difference_cleaned.csv")

# Merge with student data

student_data = pd.merge(student_data, 
                        rolling_difference_cleaned, 
                        on = 'emplid')   
    
"""
Numeric Coding Parent Education levels

There are 8 levels:

1. "No Response" 
2. "No Highschool"
3. "Some Highschool"
4. "High School Graduate"
5. "Some College"
6. "2-Year College Graduate"
7. "4-Year College Graduate"
8. "Postgraduate"

"""
def parent_ed_func(x):
    if x == "No Response":
        return 1
    elif x == "No Highschool":
        return 2
    elif x == "Some Highschool":
        return 3
    elif x == "High School Graduate":
        return 4
    elif x == "Some College":
        return 5
    elif x == "2-Year College Graduate":
        return 6
    elif x == "4-Year College Graduate":
        return 7
    return 8

student_data['parent_education_1_code'] =\
    student_data['parent_education_1'].apply(parent_ed_func)

student_data['parent_education_2_code'] =\
    student_data['parent_education_2'].apply(parent_ed_func)
    
"""
Description variable for timely graduation: 

* four_year = graduated within 4 years (8 semesters) of first enrollment
* six_year = time to graduation was more than 4 years and up to six years 
(12 semesters) of first enrollment
* DNG = student did not graduate
* other = enrolled at CSULB for longer than 6 years before graduating

"""
    
def timely_grad_func(x,y,z):
    if x == 0:
        return "Did not Graduate"
    elif z == 1 & x == 1:
        return "Summer/Winter"
    elif x == 1 & (y <= 8):
        return "Four Year Grad"
    elif x == 1 & (8 < y <= 12):
        return "Six Year Grad"
    else:
        return "other"

student_data['timely_grad'] =\
    student_data.apply(lambda x: timely_grad_func(x['graduated_flag'],
                                                  x['semesters'],
                                                  x['winter_summer_flag']),
                       axis=1)
   
    
# Check

student_data.timely_grad.unique()

"""
Binary indicator for Pell Eligible status.

Coded as 0 for "Non-Pell Eligible" and 1 for "Pell Eligible"

"""

def pell_flag_func(x):
    if x == "Non-Pell Eligible":
        return 0
    elif x == "Pell Eligible":
        return 1

student_data['pell_eligibility_coded'] =\
    student_data['pell_eligibility'].apply(pell_flag_func)
    
"""
Create a binary variable for local preference.  

This variable combines all Local status into one main "Local" variable.  
All others coded as "Non-Local"

* 1 = local
"""

def local_func(x):
    if x == "CA LOCAL LBUSD":
        return 1
    elif x == "CA LOCAL NON-LBUSD":
        return 1
    elif x == "CA NON-LOCAL":
        return 0
    elif x == "NON-U.S. HIGH SCHOOL":
        return 0
    elif x == "UNKNOWN":
        return 0
    return 0 

student_data['local_coding'] =\
    student_data['lsa_local_grouping'].apply(local_func)
    
"""
Create a binary variable for minority status.  

This variable combines all minority type statuses into one main 
"Minority" variable.  All others coded as "Non-Minority"

* 1 = minority
"""    

def minority_func(x):
    if x == "Minority":
        return 1
    return 0 

student_data['minority_coding'] =\
    student_data['minority'].apply(minority_func)   
    
"""
Adding Flag for Enrollment in a Semester
"""

enrolled_flag = pd.read_csv(r"C:\Users\yaleq\OneDrive - csulb\Yale_Quan_Thesis\Python_Code\Enrolled_Flag.csv")


student_data = pd.merge(student_data, enrolled_flag, on = 'emplid')

"""
Remove non-traditional Students
"""
# Removing Young Scholars

thesis_df = thesis_df[~thesis_df['emplid'].isin(young_scholars)]

# Removing Early Entrant Students

thesis_df = thesis_df[~thesis_df['emplid'].isin(early_entrant)]

# Removing on-degree seeking students

thesis_df = thesis_df[~thesis_df['emplid'].isin(non_degree)]

# Removing international students
thesis_df = thesis_df[~thesis_df['emplid'].isin(international)]

# Cleanup
del early_entrant
del international
del non_degree
del young_scholars

# Remove students who did not take the ACT and SAT

student_data = student_data.dropna(subset=['MAX_ACT_SAT_MATH',
                                           'MAX_ACT_SAT_Reading'])
    
# Drop rows with 100% missing

student_data = student_data.dropna(how = 'all')

# Drop columns with 100% missing
student_data = student_data.dropna(axis = 1, how = 'all')

"""
Export to CSV
"""

student_data.to_csv (r'C:\Users\yaleq\OneDrive - csulb\Yale_Quan_Thesis\student_data_2.csv', index = False, header=True)


    
    