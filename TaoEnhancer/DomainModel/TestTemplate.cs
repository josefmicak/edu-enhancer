namespace DomainModel
{
    public class TestTemplate
    {
        public string TestNameIdentifier { get; set; }
        public string TestNumberIdentifier { get; set; }
        public string Title { get; set; }
        public string NegativePoints { get; set; } = "disabled";
        public string OwnerLogin { get; set; }
        public User Owner { get; set; }
        public ICollection<QuestionTemplate> QuestionTemplateList { get; set; }
    }
}