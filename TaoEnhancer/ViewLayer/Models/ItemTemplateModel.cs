namespace ViewLayer.Models
{
    public class ItemTemplateModel : PageModel
    {
        private string testNameIdentifier;
        private string testNumberIdentifier;
        private string itemNameIdentifier;
        private string itemNumberIdentifier;
        private (string, string, string title, string label, int amountOfSubitems) itemParameters;
        private (List<string> responseIdentifierArray, List<string> responseValueArray, int errorMessageNumber) responseIdentifiers;
        private string responseIdentifier;
        private (string responseIdentifierTemp, int questionType, int subquestionPoints, bool subquestionPointsDetermined, double wrongChoicePoints, string imageSource, string subitemText, List<string> possibleAnswerArray, List<string> subquestionArray, List<string> correctChoiceArray, List<string> correctAnswerArray) subitemParameters;
        private (bool recommendedWrongChoicePoints, double selectedWrongChoicePoints, int questionPoints, bool questionPointsDetermined) questionPoints;
        private string questionTypeText;
        private bool isSelectDisabled;
        private double correctChoicePoints;
        private List<string> correctChoiceArray = new List<string>();
        private int correctAnswerCount;
        private double wrongChoicePoints;
        private int subquestionPoints;
        private string subquestionPointsText;
        private string errorText;

        public string TestNameIdentifier { get { return testNameIdentifier; } set { testNameIdentifier = value; } }
        public string TestNumberIdentifier { get { return testNumberIdentifier; } set { testNumberIdentifier = value; } }
        public string ItemNameIdentifier { get { return itemNameIdentifier; } set { itemNameIdentifier = value; } }
        public string ItemNumberIdentifier { get { return itemNumberIdentifier; } set { itemNumberIdentifier = value; } }
        public (string, string, string title, string label, int amountOfSubitems) ItemParameters { get { return itemParameters; } set { itemParameters = value; } }
        public (List<string> responseIdentifierArray, List<string> responseValueArray, int errorMessageNumber) ResponseIdentifiers { get { return responseIdentifiers; } set { responseIdentifiers = value; } }
        public string ResponseIdentifier { get { return responseIdentifier; } set { responseIdentifier = value; } }
        public (string responseIdentifierTemp, int questionType, int subquestionPoints, bool subquestionPointsDetermined, double wrongChoicePoints, string imageSource, string subitemText, List<string> possibleAnswerArray, List<string> subquestionArray, List<string> correctChoiceArray, List<string> correctAnswerArray) SubitemParameters { get { return subitemParameters; } set { subitemParameters = value; } }
        public (bool recommendedWrongChoicePoints, double selectedWrongChoicePoints, int questionPoints, bool questionPointsDetermined) QuestionPoints { get { return questionPoints; } set { questionPoints = value; } }
        public string QuestionTypeText { get { return questionTypeText; } set { questionTypeText = value; } }
        public bool IsSelectDisabled { get { return isSelectDisabled; } set { isSelectDisabled = value; } }
        public double CorrectChoicePoints { get { return correctChoicePoints; } set { correctChoicePoints = value; } }
        public List<string> CorrectChoiceArray { get { return correctChoiceArray; } set { correctChoiceArray = value; } }
        public int CorrectAnswerCount { get { return correctAnswerCount; } set { correctAnswerCount = value; } }
        public double WrongChoicePoints { get { return wrongChoicePoints; } set { wrongChoicePoints = value; } }
        public int SubquestionPoints { get { return subquestionPoints; } set { subquestionPoints = value; } }
        public string SubquestionPointsText { get { return subquestionPointsText; } set { subquestionPointsText = value; } }
        public string ErrorText { get { return errorText; } set { errorText = value; } }
    }
}