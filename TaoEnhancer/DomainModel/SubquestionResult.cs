namespace DomainModel
{
    public class SubquestionResult
    {
        public string TestResultIdentifier { get; set; }
        public string QuestionNumberIdentifier { get; set; }
        public string SubquestionIdentifier { get; set; }
        public string OwnerLogin { get; set; }
        public string[] StudentsAnswerList { get; set; }
        public double? StudentsPoints { get; set; }
        public SubquestionTemplate SubquestionTemplate { get; set; }
        public QuestionResult QuestionResult { get; set; }
    }
}