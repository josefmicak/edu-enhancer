namespace ViewLayer.Models
{
    public class BrowseSolvedItemModel : PageModel
    {
        private string testNameIdentifier;
        private string testNumberIdentifier;
        private string itemNameIdentifier;
        private string itemNumberIdentifier;
        private string deliveryExecutionIdentifier;
        private string studentIdentifier;
        private (string, string, string title, string label, int amountOfSubitems) itemParameters;
        private (List<string> responseIdentifierArray, List<string> responseValueArray, int errorMessageNumber) responseIdentifiers;
        private string responseIdentifier;
        private (string responseIdentifierTemp, int questionType, int subquestionPoints, bool subquestionPointsDetermined, double wrongChoicePoints, string imageSource, string subitemText, List<string> possibleAnswerArray, List<string> subquestionArray, List<string> correctChoiceArray, List<string> correctAnswerArray) subitemParameters;
        private List<double> studentsSubitemPointsList = new List<double>();
        private double studentsSubitemPointsListSum;
        private double studentsSubitemPoints;
        private (bool recommendedWrongChoicePoints, double selectedWrongChoicePoints, int questionPoints, bool questionPointsDetermined) questionPoints;
        private (double, List<string> studentsAnswers, string studentsAnswerCorrectLabel, string studentsAnswerPointsLabel) deliveryExecutionInfo;
        private string answerClass;
        private string questionTypeText;
        private bool isSelectDisabled;

        public string TestNameIdentifier { get { return testNameIdentifier; } set { testNameIdentifier = value; } }
        public string TestNumberIdentifier { get { return testNumberIdentifier; } set { testNumberIdentifier = value; } }
        public string ItemNameIdentifier { get { return itemNameIdentifier; } set { itemNameIdentifier = value; } }
        public string ItemNumberIdentifier { get { return itemNumberIdentifier; } set { itemNumberIdentifier = value; } }
        public string DeliveryExecutionIdentifier { get { return deliveryExecutionIdentifier; } set { deliveryExecutionIdentifier = value; } }
        public string StudentIdentifier { get { return studentIdentifier; } set { studentIdentifier = value; } }
        public (string, string, string title, string label, int amountOfSubitems) ItemParameters { get { return itemParameters; } set { itemParameters = value; } }
        public (List<string> responseIdentifierArray, List<string> responseValueArray, int errorMessageNumber) ResponseIdentifiers { get { return responseIdentifiers; } set { responseIdentifiers = value; } }
        public string ResponseIdentifier { get { return responseIdentifier; } set { responseIdentifier = value; } }
        public (string responseIdentifierTemp, int questionType, int subquestionPoints, bool subquestionPointsDetermined, double wrongChoicePoints, string imageSource, string subitemText, List<string> possibleAnswerArray, List<string> subquestionArray, List<string> correctChoiceArray, List<string> correctAnswerArray) SubitemParameters { get { return subitemParameters; } set { subitemParameters = value; } }
        public List<double> StudentsSubitemPointsList { get { return studentsSubitemPointsList; } set { studentsSubitemPointsList = value; } }
        public double StudentsSubitemPointsListSum { get { return studentsSubitemPointsListSum; } set { studentsSubitemPointsListSum = value; } }
        public double StudentsSubitemPoints { get { return studentsSubitemPoints; } set { studentsSubitemPoints = value; } }
        public (bool recommendedWrongChoicePoints, double selectedWrongChoicePoints, int questionPoints, bool questionPointsDetermined) QuestionPoints { get { return questionPoints; } set { questionPoints = value; } }
        public (double, List<string> studentsAnswers, string studentsAnswerCorrectLabel, string studentsAnswerPointsLabel) DeliveryExecutionInfo { get { return deliveryExecutionInfo; } set { deliveryExecutionInfo = value; } }
        public string AnswerClass { get { return answerClass; } set { answerClass = value; } }
        public string QuestionTypeText { get { return questionTypeText; } set { questionTypeText = value; } }
        public bool IsSelectDisabled { get { return isSelectDisabled; } set { isSelectDisabled = value; } }
    }
}