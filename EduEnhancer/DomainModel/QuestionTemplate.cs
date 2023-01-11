namespace DomainModel
{
    public class QuestionTemplate
    {
        public string QuestionNameIdentifier { get; set; } = default!;
        public string QuestionNumberIdentifier { get; set; } = default!;
        public string Title { get; set; } = default!;
        public string Label { get; set; } = default!;
        public string OwnerLogin { get; set; } = default!;
        public TestTemplate TestTemplate { get; set; } = default!;
        public ICollection<SubquestionTemplate> SubquestionTemplateList { get; set; } = default!;
    }
}