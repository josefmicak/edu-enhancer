using DomainModel;

namespace ViewLayer.Models
{
    public class BrowseSolvedQuestionModel : PageModel
    {
        private QuestionTemplate questionTemplate;
        private QuestionResult questionResult;
        private string testNameIdentifier;
        private string testNumberIdentifier;
        private string testResultIdentifier;
        private SubquestionTemplate subquestionTemplate;
        private SubquestionResult subquestionResult;
        private string[] subquestionTypeTextArray;

        public QuestionTemplate QuestionTemplate { get { return questionTemplate; } set { questionTemplate = value; } }
        public QuestionResult QuestionResult { get { return questionResult; } set { questionResult = value; } }
        public string TestNameIdentifier { get { return testNameIdentifier; } set { testNameIdentifier = value; } }
        public string TestNumberIdentifier { get { return testNumberIdentifier; } set { testNumberIdentifier = value; } }
        public string TestResultIdentifier { get { return testResultIdentifier; } set { testResultIdentifier = value; } }
        public SubquestionTemplate SubquestionTemplate { get { return subquestionTemplate; } set { subquestionTemplate = value; } }
        public SubquestionResult SubquestionResult { get { return subquestionResult; } set { subquestionResult = value; } }
        public string[] SubquestionTypeTextArray { get { return subquestionTypeTextArray; } set { subquestionTypeTextArray = value; } }
    }
}
