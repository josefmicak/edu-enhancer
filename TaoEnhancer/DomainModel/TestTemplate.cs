namespace DomainModel
{
    public class TestTemplate
    {
        public string TestNameIdentifier { get; set; } = default!;
        public string TestNumberIdentifier { get; set; } = default!;
        public string Title { get; set; } = default!;
        public string NegativePoints { get; set; } = "disabled";
        public string OwnerLogin { get; set; } = default!;
        public User Owner { get; set; } = default!;
        public ICollection<QuestionTemplate> QuestionTemplateList { get; set; } = default!;
    }
}