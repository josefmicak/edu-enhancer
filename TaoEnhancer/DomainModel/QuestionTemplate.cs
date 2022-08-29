namespace DomainModel
{
    public class QuestionTemplate
    {
        public string QuestionNameIdentifier { get; set; }
        public string QuestionNumberIdentifier { get; set; }
        public string Title { get; set; }
        public string Label { get; set; }
        public string OwnerLogin { get; set; }
        public TestTemplate TestTemplate { get; set; }
        public ICollection<SubquestionTemplate> SubquestionTemplateList { get; set; }
    }
}