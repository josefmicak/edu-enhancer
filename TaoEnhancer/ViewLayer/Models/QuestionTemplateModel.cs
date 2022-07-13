using DomainModel;

namespace ViewLayer.Models
{
    public class QuestionTemplateModel : PageModel
    {
        private QuestionTemplate selectedQuestionTemplate;
        private string testNameIdentifier;
        private string testNumberIdentifier;
        private string[] subquestionTypeTextArray;
        private SubquestionTemplate selectedSubquestionTemplate;

        public QuestionTemplate SelectedQuestionTemplate { get { return selectedQuestionTemplate; } set { selectedQuestionTemplate = value; } }
        public string TestNameIdentifier { get { return testNameIdentifier; } set { testNameIdentifier = value; } }
        public string TestNumberIdentifier { get { return testNumberIdentifier; } set { testNumberIdentifier = value; } }
        public string[] SubquestionTypeTextArray { get { return subquestionTypeTextArray; } set { subquestionTypeTextArray = value; } }
        public SubquestionTemplate SelectedSubquestionTemplate { get { return selectedSubquestionTemplate; } set { selectedSubquestionTemplate = value; } }
    }
}
