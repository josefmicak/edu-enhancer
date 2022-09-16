namespace DomainModel
{
    public class SubquestionResult
    {
        public string TestResultIdentifier { get; set; } = default!;
        public string QuestionNumberIdentifier { get; set; } = default!;
        public string SubquestionIdentifier { get; set; } = default!;
        public string OwnerLogin { get; set; } = default!;
        public string[] StudentsAnswerList { get; set; } = default!;//null?
        public double? StudentsPoints { get; set; } = default!;
        public SubquestionTemplate SubquestionTemplate { get; set; } = default!;
        public QuestionResult QuestionResult { get; set; } = default!;
    }
}